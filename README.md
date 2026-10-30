# AI Customer Support Agent

An AI-powered customer support application built with Python, Streamlit, and Ollama. The application helps customers find answers to common questions, check sample order details, and submit support requests through an interactive chatbot.

It also includes customer feedback collection, support ticket management, and chat history downloads.

## Features

### 1. AI Customer Support Chatbot
- Answer customer questions using a local LLM.
- Provide responses based on the store's FAQ knowledge base.
- Categorize customer queries.
- Retrieve relevant FAQs using TF-IDF and cosine similarity.
- Handle unsupported questions by recommending human support.

### 2. Customer Feedback
- Collect feedback on AI-generated responses.
- Allow customers to mark answers as helpful or not helpful.
- Accept additional customer comments.
- Save feedback locally in a CSV file.
- View feedback and satisfaction statistics.

### 3. Support Ticket Management
- Automatically create tickets for selected escalated complaints.
- Generate unique ticket IDs.
- Assign ticket priorities.
- Track ticket status: Open, In Progress, and Resolved.
- Update and save ticket information.
- Download support ticket records.

### 4. Order Tracking Demo
- Search orders using an order ID.
- Display customer and product information.
- View sample order statuses.
- Display estimated delivery dates.
- Handle invalid order IDs.

### 5. Better Chat Experience
- Interactive Streamlit chat interface.
- Suggested questions for customers.
- Chat history during the session.
- Loading indicator while generating responses.
- Clear conversation option.
- Download chat history as a text file.

## Technology Stack

- **Programming Language:** Python
- **Frontend:** Streamlit
- **AI Model:** Llama 3.2 (3B)
- **LLM Integration:** Ollama
- **Text Retrieval:** Scikit-learn
- **Similarity Search:** TF-IDF and Cosine Similarity
- **Data Processing:** Pandas
- **Data Storage:** CSV and JSON

## Project Structure

```text
ai-customer-support-agent/
│
├── app.py
├── support_logic.py
├── knowledge_base.json
├── orders.csv
├── requirements.txt
├── README.md
├── .gitignore
│
├── tickets.csv       # Generated locally
└── feedback.csv      # Generated locally
```

## Installation and Setup

### Prerequisites

- Python 3
- Visual Studio Code or another Python editor
- Ollama installed
- Llama 3.2 (3B) model

### 1. Clone the Repository

```bash
git clone https://github.com/Abhishek-Gosain/ai-customer-support-agent.git
cd ai-customer-support-agent
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Set Up Ollama

Install Ollama from:

https://ollama.com

Download the model:

```bash
ollama pull llama3.2:3b
```

Make sure Ollama is running before starting the application.

### 5. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal. The default address is:

```text
http://localhost:8501
```

## How to Use

1. Open the application in your browser.
2. Start a conversation with the AI support assistant.
3. Ask questions about store policies and services.
4. Use the suggested questions for quick responses.
5. Submit feedback on chatbot answers.
6. Search for sample orders using their order IDs.
7. View escalated requests in the Support Tickets section.
8. Update ticket statuses from the admin view.
9. Download chat history, feedback, and ticket records.

## Sample Order IDs

The project includes fictional order records for testing.

| Order ID | Product | Status |
|---|---|---|
| 1001 | Laptop | Shipped |
| 1002 | Wireless Mouse | Delivered |
| 1003 | Keyboard | Processing |
| 1004 | Monitor | Out for Delivery |
| 1005 | Headphones | Cancelled |
| 1006 | Smartphone | Shipped |
| 1007 | Tablet | Processing |

## Data Storage

The application uses local files:

- `knowledge_base.json` stores FAQs and store policies.
- `orders.csv` contains fictional order records.
- `tickets.csv` stores generated support tickets.
- `feedback.csv` stores customer ratings and comments.

The ticket and feedback files are generated automatically when required.

## Important Notes

- This project uses fictional store and order information for demonstration purposes.
- Order tracking is based on the included sample CSV file and does not connect to a real delivery service.
- Support tickets are stored locally and do not notify real support agents.
- Customer feedback and ticket records are stored in local CSV files.
- The application does not currently include authentication or role-based access.
- AI-generated answers should be checked before being used for actual customer support.

## Future Improvements

- Semantic search using embeddings and a vector database.
- Multilingual customer support.
- User authentication and admin access.
- Database integration for persistent storage.
- Live order tracking through external APIs.
- Email notifications for support tickets.
- Deployment for online access.

## Author

**Abhishek Gosain**

GitHub: https://github.com/Abhishek-Gosain

## License

This project is intended for learning and portfolio demonstration. Add a license file if you plan to distribute or reuse it under specific terms.
