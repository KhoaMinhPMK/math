from flask import Flask, request, jsonify
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import warnings
import pickle
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from flask_cors import CORS

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Directly assign the API key (for testing purposes only)
GOOGLE_API_KEY = 'AIzaSyBalL60cVNzIqWR8TstxSeM2ytq9_h1Yz4'
# Ensure the API key is set in the environment variables for security
# GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
# if not GOOGLE_API_KEY:
#     raise ValueError("The Google API Key must be set in environment variables.")

# Configure the generative AI with the API key
genai.configure(api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

PROCESSED_FILE = "processed_pdf_data.pkl"

def load_processed_data(filename):
    """Load processed data from a file if it exists."""
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    else:
        return None

def get_qa_chain():
    """Create and return the QA chain."""
    processed_data = load_processed_data(PROCESSED_FILE)
    if processed_data:
        texts, embedded_texts = processed_data

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = Chroma.from_texts(texts, embeddings)
    vector_index = vector_store.as_retriever(search_kwargs={"k": 5})
    model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, temperature=0.8, convert_system_message_to_human=True)
    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=vector_index,
        return_source_documents=False
    )
    return qa_chain

qa_chain = get_qa_chain()

template = """
bạn là một chat bot tên fispage chuyên gia tạo bài tập cho học sinh trong môn toán.Dùng các đoạn văn bản sau đây để trả lời yêu cầu của người dùng một cách thật chi tiết. Dịch câu hỏi của người dùng sang tiếng Anh trước khi trả lời. Bạn chỉ trả lời các tin nhắn liên quan đến toán lớp 9.
Khi người dùng cần thay đổi dạng bài tập. Hãy thay đổi nội dung sao cho các bài có nhiều điểm khác nhau nhất có thể nhưng vẫn đúng dạng người dùng yêu cầu

Các câu hỏi có độ khó và phức tạp cao.
khi thấy có những cụm từ liên quan đến tạo bài tương tự hoặc tạo bài tập giống như này tức là bạn phải tạo ra những bài tập cùng dạng nhưng khác chủ đề so với bài tập người dùng gửi vào. Và tạo theo đúng số lượng mà người dùng yêu cầu và giải chúng thật chi tiết.
ví dụ như sẽ có yêu cầu mà người dùng gửi như sau: tạo bài tập tương tự như dạng này cho tôi Hai thành phố A và B cách nhau 100km. Cùng một lúc, hai xe chuyển động đều ngược chiều nhau, xe ô tô đi từ A với vận tốc 30km/h, xe mô tô đi từ B với vận tốc 20km/h. Chọn A làm gốc toạ độ, chiều dương từ A đến B, gốc thời gian là lúc hai xe bắt đầu đi.
thì lúc này bạn sẽ nhận biết xem đây là dạng bài gì, ví dụ như đây là dạng bài toán chuyển động. Sau khi nhận diện xong bạn sẽ tạo ra bài toán chuyển động tương tự như bài người dùng vừa gửi vô nhưng phải thay đổi chủ đề, độ khó, số liệu, ngữ cảnh.

Khi tạo ra các bài toán lớp 9 theo yêu cầu của người dùng, đảm bảo rằng các bài toán và lời giải chi tiết, rõ ràng và dễ hiểu. Nếu gặp câu hỏi không liên quan đến hoặc vượt quá phạm vi của toán lớp 9, trả lời "Xin lỗi, tôi không được đào tạo để xử lý câu hỏi này." Nếu bạn không biết câu trả lời, trả lời "Xin lỗi, tôi không biết" và không cố gắng chế tạo câu trả lời. Việc đảm bảo độ chính xác và phù hợp với kiến thức của học sinh trung học cơ sở là rất quan trọng. Khi trả lời, cũng cung cấp lời giải và ví dụ minh họa để giúp người dùng hiểu rõ hơn. Nếu ai đó hỏi tên bạn, trả lời rằng tên bạn là FisMath. Ngoài ra, giữ giọng điệu thân thiện và hỗ trợ, khuyến khích người dùng học và nâng cao kiến thức toán học của họ.

Định dạng câu trả lời của bạn sử dụng HTML để cải thiện tính đọc. Ví dụ, sử dụng <b> cho văn bản in đậm, <i> cho văn bản in nghiêng, <ul> cho danh sách gạch đầu dòng và <ol> cho danh sách đánh số.
"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

previous_message = None
state = False  # Initialize state to False

def custom_qa_chain(query):
    global previous_message
    global state
    
    # Define relevant math keywords
    math_keywords = ["math", "toán", "bài tập", "homework", "bài"]
    dif_sub = ["vật lý", "hóa học", "sinh học", "ngữ văn", "lịch sử", "địa lý", "giáo dục công dân", "công nghệ", "tin học", "tiếng anh"]

    # Check if the query is relevant to math problems
    if not any(keyword in query.lower() for keyword in math_keywords) and not state:
        return "Xin chào bạn, tôi là chatbot tạo bài tập toán. Tôi không có khả năng phản hồi các thông tin khác. Nếu cần tạo bài tập hãy nhập yêu cầu: 'VD: tạo cho tôi 5 bài toán về tính tỉ lệ phần trăm'"

    if any(keyword in query.lower() for keyword in dif_sub) and not state:
        return "Xin lỗi bạn, tôi chỉ có thể tạo được môn toán"
    
    if any(keyword in query.lower() for keyword in math_keywords):
        # Handle creating math problems
        if "tạo" in query.lower() and any(keyword in query.lower() for keyword in math_keywords) and not state:
            if "dạng" in query.lower():
                sentence = query.lower()
                index = sentence.index("dạng") + len("dạng")
                remaining_text = sentence[index:].strip()
                remaining_words = remaining_text.split()
                num_words = len(remaining_words)
                if any(keyword in query.lower() for keyword in ["tương tự", "giống", "tương ứng"]) and len(query.lower()) <= 50 and "hỏi" not in query.lower():
                    state = True  # Set state to True when "tương tự" or "giống" is found
                    previous_message = query.lower()
                    return "Hãy cho tôi bài tập mà bạn muốn tạo giống để có thể hỗ trợ bạn tốt hơn"
                elif not any(keyword in query.lower() for keyword in ["tương tự", "giống", "tương ứng"]) and num_words <= 3:
                    return "Hãy cho tôi biết dạng bạn muốn tạo để có thể hỗ trợ bạn tốt hơn"
            elif "dạng" not in query.lower():
                if any(keyword in query.lower() for keyword in ["tương tự", "giống", "tương ứng"]) and len(query.lower()) <= 50 and "hỏi" not in query.lower():
                    state = True  # Set state to True when "tương tự" or "giống" is found
                    previous_message = query.lower()
                    return "Hãy cho tôi bài tập mà bạn muốn tạo giống để có thể hỗ trợ bạn tốt hơn"
    

    # Handle follow-up for creating similar problems
    if state:
        query = "Hãy giả sử bạn chính là một chuyên gia toán học và thay đổi nội dung và đề tài nhiều nhất có thể để tạo ra bài tập có dạng giống với bài sau. Bạn không được đưa đáp án cho bài của tôi: " + query
        previous_message = None  # Reset after combining
        state = False  # Reset state
        result = qa_chain({"query": query})
        return format_response(result["result"])

    # Process the query using the QA chain
    try:
        result = qa_chain({"query": query})
        return format_response(result["result"])
    except Exception as e:
        return f"Sorry, an error occurred: {str(e)}"
def format_response(response):
    """Format the response for better readability."""
    if "không biết" in response.lower() or "không thể" in response.lower() or response.lower() == None:
        return "Bạn có thể cho tôi chi tiết hơn về bài bạn muốn tạo"

    formatted_response = response.replace("\n", "<br>").replace("**", "<b>").replace("```", "<pre>").replace("**", "</b>")
    return formatted_response

@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint to handle user queries."""
    data = request.json
    query = data.get("message", "")
    response = custom_qa_chain(query)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
