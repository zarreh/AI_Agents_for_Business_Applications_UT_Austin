# Agentic AI for Business Applications

A comprehensive learning repository for the **UT Austin × Great Learning** course: *"Agentic AI for Business Applications"*. This course teaches you how to build intelligent LLM-powered agents and Retrieval-Augmented Generation (RAG) systems with real-world business applications.

## 🎯 Course Overview

This repository contains hands-on Jupyter notebooks structured around two main learning tracks:

### Week 1: Retrieval-Augmented Generation (RAG)
- Build a **Medical Assistant** system that answers healthcare questions by retrieving information from Merck Manuals
- Learn prompt engineering, embeddings, and vector databases
- Understand document chunking, retrieval strategies, and evaluation metrics

### Project 1: Multi-Agent Business Systems
- Implement agents that tackle **Apple's organizational innovation** challenge
- Learn multi-agent coordination and task decomposition
- Apply RAG techniques to real business case studies

## 🚀 Quick Start

### 1. Prerequisites
- Python ≥ 3.10
- `uv` package manager (already configured)
- Jupyter Notebook or JupyterLab

### 2. Clone the Repository
```bash
git clone https://github.com/zarreh/AI_Agents_for_Business_Applications_UT_Austin.git
cd AI_Agents_for_Business_Applications_UT_Austin
```

### 3. Activate the Environment
```bash
source .venv/bin/activate
```

If the environment doesn't exist, create it:
```bash
uv sync
```

### 4. Start Jupyter
```bash
jupyter notebook
# or
jupyter lab
```

### 5. Select the Kernel
When opening a notebook:
1. Click **Kernel** → **Change kernel**
2. Select **Python (ai_biz_ut_austin)**

## 📁 Project Structure

```
agnetic_ai_foundation/
├── week1/                                    # RAG & Prompt Engineering Track
│   ├── MLS_Week_2_Medical_Assistant.ipynb   # Complete RAG system (71 cells)
│   ├── lecture_notebooks/
│   │   └── Hands_on_RAG.ipynb               # Interactive RAG patterns
│   ├── extra/
│   │   └── LLM_Powered_Research_Assistant.ipynb
│   ├── lectures/                            # PDF slides and notes
│   └── medical_diagnosis_manual.pdf         # Reference document
│
└── project1/                                 # Business Agent Track
    ├── Learners_Notebook_Full_Code.ipynb    # Complete solution
    ├── Learners_Notebook_Low_Code.ipynb     # Scaffolded version (try this first!)
    ├── Apple_HBR_Report_Project_Template.pptx
    └── HBR_How_Apple_Is_Organized_For_Innovation.pdf
```

## 🔧 Technology Stack

| Component | Technology |
|-----------|-----------|
| **LLM Framework** | LangChain 0.3.27 + langchain-community |
| **LLM Provider** | OpenAI API (custom endpoint) |
| **Vector Database** | ChromaDB 1.5.9 (primary), FAISS 1.14.2 (fallback) |
| **Embeddings** | sentence-transformers 5.5.1 |
| **Document Processing** | PyMuPDF, LangChain TextSplitter |
| **Data Tools** | pandas, Hugging Face Datasets |
| **Tracing & Debugging** | LangSmith |

## 📚 Getting Started with Notebooks

### For Beginners
1. Start with **Week 1**: Open `week1/MLS_Week_2_Medical_Assistant.ipynb`
2. Run cells sequentially from top to bottom
3. Read the markdown cells—they contain important context and business problems

### For Project Work
1. Open `project1/Learners_Notebook_Low_Code.ipynb` (scaffolded version)
2. Complete the code cells with guidance from comments
3. Compare with `Learners_Notebook_Full_Code.ipynb` to see complete solutions

## 🔐 Configuration

This project uses a `config.json` file for API credentials:

```json
{
  "OPENAI_API_BASE": "https://your-endpoint.com/openai/v1",
  "API_KEY": "your-api-key",
  "LANGSMITH_API_KEY": "your-langsmith-key",
  "LANGSMITH_ENDPOINT": "https://api.smith.langchain.com",
  "TAVILY_API_KEY": "your-tavily-key"
}
```

**Important**: `config.json` is in `.gitignore` and not tracked in version control. Get your credentials from:
- OpenAI/GreatLearning custom endpoint
- LangSmith dashboard
- Tavily (search API)

## 🛠️ Common Tasks

### Add New Dependencies
```bash
source .venv/bin/activate
uv add <package_name>
uv sync  # Update environment
```

### View Dependency Tree
```bash
uv tree
```

### Export Requirements
```bash
uv export --format requirements.txt -o requirements.txt
```

### Restart Jupyter Kernel
If you encounter import errors after installing packages:
1. **Kernel** → **Restart & Run All Cells**
2. Or from terminal: `jupyter notebook --NotebookApp.notebook_dir=.`

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'langchain'"
```bash
deactivate
source .venv/bin/activate
python -m pip list | grep langchain
```

### "No kernel named 'ai_biz_ut_austin' found"
```bash
python -m ipykernel install --user --name ai_biz_ut_austin --display-name "Python (ai_biz_ut_austin)"
```

### OpenAI API Errors
1. Verify `config.json` is in the project root
2. Check that `OPENAI_API_BASE` and `API_KEY` are correct
3. Test the connection:
   ```python
   from openai import OpenAI
   client = OpenAI(api_key=config["API_KEY"], base_url=config["OPENAI_API_BASE"])
   ```

### ChromaDB Persistence Issues
By default, ChromaDB stores vectors in memory. For persistent storage:
```python
from langchain.vectorstores import Chroma
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
```

## 📖 Learning Path

| Week | Topic | Notebook | Focus |
|------|-------|----------|-------|
| 1 | RAG Fundamentals | `MLS_Week_2_Medical_Assistant.ipynb` | PDF loading, chunking, embeddings, retrieval |
| 1 | Hands-on RAG | `Hands_on_RAG.ipynb` | Interactive patterns and best practices |
| Project 1 | Business Agents | `Learners_Notebook_Low_Code.ipynb` | Agent coordination, evaluation metrics |
| Project 1 | Advanced | `Learners_Notebook_Full_Code.ipynb` | Complete multi-agent system |

## 📚 Key Concepts

### RAG Pipeline
1. **Load**: Extract text from PDFs using `PyMuPDFLoader`
2. **Chunk**: Split into overlapping segments with `RecursiveCharacterTextSplitter`
3. **Embed**: Generate vectors using `SentenceTransformerEmbeddings`
4. **Store**: Index vectors in `ChromaDB`
5. **Retrieve**: Query relevant documents for context
6. **Generate**: Use LLM to answer based on retrieved context
7. **Evaluate**: Measure accuracy, relevance, and business metrics

### Agent Pattern
```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4"),
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)
result = qa.run("Your question here")
```

## 🔗 Resources

- **LangChain Documentation**: https://python.langchain.com/
- **ChromaDB Guide**: https://docs.trychroma.com/
- **OpenAI API**: https://platform.openai.com/docs/
- **Course Materials**: Available on Great Learning platform
- **Lecture Slides**: See `agnetic_ai_foundation/week1/lectures/`

## 🤝 Contributing

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Commit: `git commit -m "Add your message"`
4. Push: `git push origin feature/your-feature`
5. Open a pull request

## 📝 License

This educational material is provided by UT Austin and Great Learning. Refer to course materials for usage rights.

## 👥 Authors

- Course instructors: UT Austin × Great Learning
- Repository maintained by: AI development team

## 📞 Support

For course-related questions:
- **Great Learning Platform**: Available in your course dashboard
- **UT Austin Resources**: Check course portal

For technical issues:
- Review the **Troubleshooting** section above
- Check notebook markdown cells for hints
- Examine `AGENTS.md` for developer-focused guidance

---

**Last Updated**: May 23, 2026  
**Python Version**: 3.12.10  
**LangChain Version**: 0.3.27  
**Package Manager**: uv (latest)

Happy learning! 🚀
