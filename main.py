import os
import json
from typing import List, Optional

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.chunking import HybridChunker

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

BID_DIRECTORIES = ["Bid2"]
OUTPUT_DIR = "./output"
CHROMA_DB_PATH = "./chroma_db"
SUPPORTED_FORMATS = ['.pdf', '.html', '.htm']
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
GEMINI_MODEL = "gemini-2.5-pro"

class BidInformation(BaseModel):
    bid_number: Optional[str] = None
    title: Optional[str] = None
    due_date: Optional[str] = None
    bid_submission_type: Optional[str] = None
    term_of_bid: Optional[str] = None
    pre_bid_meeting: Optional[str] = None
    installation: Optional[str] = None
    bid_bond_requirement: Optional[str] = None
    delivery_date: Optional[str] = None
    payment_terms: Optional[str] = None
    additional_documentation: Optional[str] = None
    mfg_for_registration: Optional[str] = None
    contract_cooperative: Optional[str] = None
    model_no: Optional[str] = None
    part_no: Optional[str] = None
    product: Optional[str] = None
    contact_info: Optional[str] = None
    company_name: Optional[str] = None
    bid_summary: Optional[str] = None
    product_specification: Optional[str] = None

def setup_docling_converter():
    return DocumentConverter(
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.HTML,
        ],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=PdfPipelineOptions(
                    do_ocr=True,
                    do_table_structure=True,
                    do_underfull_bbox_heuristic=True,
                    extract_images_in_pdf=True,
                    extract_embedded_files=True,
                ),
            ),
        },
    )

def setup_chromadb():
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    
    collection = client.get_or_create_collection(
        name="bid_documents",
        metadata={"description": "Bid documents for RAG pipeline"}
    )
    
    return client, collection

def setup_embedding_model():
    print("🔧 Loading BAAI embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"✅ Embedding model loaded: {EMBEDDING_MODEL}")
    return model

def setup_langchain_components():
    print("🔧 Setting up LangChain components...")
    
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.1,
        max_tokens=2048,
        response_mime_type="application/json"
    )
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    
    parser = PydanticOutputParser(pydantic_object=BidInformation)
    
    print("✅ LangChain components ready")
    return llm, embeddings, parser

def get_bid_files(bid_directories: List[str]) -> List[str]:
    files = []
    for directory in bid_directories:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path):
                    file_ext = os.path.splitext(file)[1].lower()
                    if file_ext in SUPPORTED_FORMATS:
                        files.append(file_path)
    return files

def process_document(file_path: str, converter, chunker):
    print(f"Processing: {file_path}")
    
    try:
        result = converter.convert(file_path)
        document = result.document
        chunks = list(chunker.chunk(document))
        
        print(f"✓ Created {len(chunks)} chunks from {os.path.basename(file_path)}")
        return chunks, document
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {str(e)}")
        return None, None

def extract_text_from_document(document) -> str:
    if not document:
        return ""
    
    text_parts = []
    
    if hasattr(document, 'text') and document.text:
        text_parts.append(document.text)
    
    if hasattr(document, 'iterate_items'):
        for item in document.iterate_items():
            if hasattr(item, 'text') and item.text:
                text_parts.append(item.text)
    
    return "\n".join(text_parts)

def store_chunks_in_chromadb(chunks, file_path: str, collection, embedding_model):
    if not chunks:
        return
    
    texts = [chunk.text for chunk in chunks]
    metadatas = [{"source_file": file_path, "chunk_index": i} for i in range(len(chunks))]
    ids = [f"{os.path.basename(file_path)}_{i}" for i in range(len(chunks))]
    
    print(f"🔢 Generating embeddings for {len(texts)} chunks...")
    embeddings = embedding_model.encode(texts).tolist()
    
    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings
    )
    
    print(f"✅ Stored {len(chunks)} chunks in ChromaDB")

def create_extraction_chain(llm, parser):
    prompt_template = """
    You are an expert at extracting structured information from bid documents.
    
    Based on the following context from bid documents, extract the required information and format it as a JSON object.
    
    Context:
    {context}
    
    Instructions:
    - Carefully read through the entire context to find information for each field
    - Look for bid numbers, titles, dates, contact information, product details, etc.
    - Extract information for each field based on the context provided
    - If information is not found, set the field to null
    - Be precise and accurate in your extraction
    - Return ONLY a valid JSON object with the following structure
    - Do not include any explanatory text, just the JSON
    
    Important: You must extract information for ALL fields. Look carefully through the entire document for:
    - Bid numbers (RFP, solicitation, contract numbers)
    - Titles and descriptions
    - Due dates and deadlines
    - Contact information and company names
    - Product specifications and model numbers
    - Payment terms and requirements
    
    {format_instructions}
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    extraction_chain = prompt | llm | parser
    return extraction_chain

def parse_llm_output(llm_output, parser):
    import re
    import json
    
    if hasattr(llm_output, 'content'):
        text = llm_output.content
    else:
        text = str(llm_output)
    
    print(f"🔍 Raw LLM output: {text[:200]}...")
    
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            json_data = json.loads(json_str)
            return BidInformation(**json_data)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"❌ JSON parsing failed: {e}")
            print(f"JSON string: {json_str}")
    
    print("⚠️ Using fallback: creating BidInformation with null values")
    return BidInformation()

def extract_bid_information_rag(directory: str, llm, embeddings, parser):
    print(f"\n🔍 Extracting bid information from {directory} using multi-request approach...")
    
    file_paths = get_bid_files([directory])
    if not file_paths:
        print(f"❌ No files found in {directory}")
        return None
    
    converter = setup_docling_converter()
    chunker = HybridChunker()
    
    all_chunks = []
    for file_path in file_paths:
        chunks, document = process_document(file_path, converter, chunker)
        if chunks:
            all_chunks.extend(chunks)
    
    if not all_chunks:
        print(f"❌ No chunks created from {directory}")
        return None
    
    from langchain_core.documents import Document
    langchain_docs = []
    for i, chunk in enumerate(all_chunks):
        text_content = str(chunk.text) if hasattr(chunk, 'text') else str(chunk)
        langchain_docs.append(
            Document(
                page_content=text_content, 
                metadata={"source": f"chunk_{i}", "chunk_index": i}
            )
        )
    
    vectorstore = Chroma.from_documents(
        documents=langchain_docs,
        embedding=embeddings,
        persist_directory=f"{CHROMA_DB_PATH}_{directory}"
    )
    
    print(f"✅ Created vectorstore with {len(langchain_docs)} chunks")
    
    query_groups = [
        {
            "name": "Phase 1 - Basic Information",
            "query": "bid number title company name contact information due date deadline closing date submission deadline response due proposal due bid due closing time submission date response date submission type method format electronic paper contact person point of contact representative phone email address",
            "fields": ["bid_number", "title", "company_name", "contact_info", "due_date", "bid_submission_type"]
        },
        {
            "name": "Phase 1 - Product Overview", 
            "query": "product item equipment service solution description model number model no product model part number part no component number specification technical details requirements specs specifications technical specifications product code item number serial number",
            "fields": ["product", "model_no", "part_no", "product_specification"]
        },
        {
            "name": "Phase 2 - Contract Terms",
            "query": "term duration period validity contract length payment terms schedule billing invoicing invoice payment delivery date shipment completion timeline installation setup implementation deployment services requirements bond requirements bid bond security deposit guarantee performance bond",
            "fields": ["term_of_bid", "payment_terms", "delivery_date", "installation", "bid_bond_requirement"]
        },
        {
            "name": "Phase 2 - Process Requirements",
            "query": "pre bid meeting conference information session mandatory meeting pre-bid conference documentation required supporting documents attachments additional documentation manufacturer registration vendor registration mfg registration cooperative agreement type vehicle contract cooperative summary executive summary overview description",
            "fields": ["pre_bid_meeting", "additional_documentation", "mfg_for_registration", "contract_cooperative", "bid_summary"]
        },
        {
            "name": "Phase 3 - Refinement Search",
            "query": "missing fields not found due date deadline contact information model number part number pre bid meeting installation bond requirements additional documentation manufacturer registration",
            "fields": ["due_date", "contact_info", "model_no", "part_no", "pre_bid_meeting", "installation", "bid_bond_requirement", "additional_documentation", "mfg_for_registration"]
        }
    ]
    
    extracted_data = {field: None for field in BidInformation.__fields__.keys()}
    
    for i, group in enumerate(query_groups):
        print(f"\n🔍 Request {i+1}/5: {group['name']}...")
        
        import time
        if i > 0:
            time.sleep(3)
        
        try:
            relevant_docs = vectorstore.similarity_search(group["query"], k=12)
            
            if not relevant_docs:
                print(f"❌ No relevant chunks found for {group['name']}")
                continue
            
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            print(f"📄 Processing {len(context)} characters for {group['name']}...")
            
            group_prompt = f"""
            You are an expert at extracting structured information from bid documents.
            
            Based on the following context, extract ONLY the specified fields and return them as a JSON object.
            
            Context:
            {context}
            
            Instructions:
            - Extract information for ONLY these fields: {', '.join(group['fields'])}
            - Look for various phrasings and synonyms (e.g., "due date" could be "deadline", "closing date", "submission deadline")
            - If information is not found, set the field to null
            - Be precise and accurate in your extraction
            - Return ONLY a valid JSON object with the specified fields
            - Do not include any explanatory text, just the JSON
            
            Required JSON structure:
            {{
                {', '.join([f'"{field}": "extracted value or null"' for field in group['fields']])}
            }}
            """
            
            response = llm.invoke(group_prompt)
            response_text = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            
            print(f"🔍 Response: {response_text[:150]}...")
            
            import json
            import re
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    group_data = json.loads(json_str)
                    for field, value in group_data.items():
                        if value and value != "null" and value.strip():
                            if extracted_data[field] is None:
                                extracted_data[field] = value
                                print(f"✅ {field}: {str(value)[:50]}...")
                            else:
                                print(f"🔄 {field}: Already found, keeping previous value")
                        else:
                            if extracted_data[field] is None:
                                print(f"❌ {field}: Not found")
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"❌ JSON parsing failed for {group['name']}: {e}")
            else:
                print(f"❌ No JSON found in response for {group['name']}")
            
        except Exception as e:
            print(f"❌ Error processing {group['name']}: {e}")
    
    try:
        result = BidInformation(**extracted_data)
        print(f"\n✅ Extraction completed with {sum(1 for v in extracted_data.values() if v)}/{len(extracted_data)} fields found")
        return result
    except Exception as e:
        print(f"❌ Error creating BidInformation: {e}")
        return None

def process_directory_with_rag(directory: str):
    print(f"🚀 Processing {directory} with RAG")
    print("=" * 50)
    
    llm, embeddings, parser = setup_langchain_components()
    bid_info = extract_bid_information_rag(directory, llm, embeddings, parser)
    
    if bid_info:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        results = {
            "directory": directory,
            "extraction_method": "RAG with LangChain + Gemini",
            "model": GEMINI_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "extracted_data": bid_info.model_dump()
        }
        
        output_file = os.path.join(OUTPUT_DIR, f"{directory}_rag_extraction.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Results saved to: {output_file}")
        
        print(f"\n📊 EXTRACTED INFORMATION FROM {directory}:")
        print("=" * 50)
        for field, value in bid_info.model_dump().items():
            status = "✅" if value else "❌"
            display_value = str(value)[:100] + "..." if value and len(str(value)) > 100 else value
            print(f"{status} {field}: {display_value}")
        
        return bid_info
    else:
        print(f"❌ Failed to extract information from {directory}")
        return None

def main():
    print("🚀 Starting RAG-based Bid Document Processing")
    print("=" * 55)
    
    all_results = {}
    
    for directory in BID_DIRECTORIES:
        if os.path.exists(directory):
            result = process_directory_with_rag(directory)
            if result:
                all_results[directory] = result.model_dump()
        else:
            print(f"⚠️ Directory {directory} not found, skipping...")
    
    if all_results:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        combined_results = {
            "summary": {
                "total_directories_processed": len(all_results),
                "extraction_method": "RAG with LangChain + Gemini",
                "model": GEMINI_MODEL,
                "embedding_model": EMBEDDING_MODEL
            },
            "extracted_data": all_results
        }
        
        output_file = os.path.join(OUTPUT_DIR, "combined_rag_extraction.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Combined results saved to: {output_file}")
        print("\n🎉 RAG-based extraction completed successfully!")
    else:
        print("❌ No directories processed successfully")

if __name__ == "__main__":
    main()
    