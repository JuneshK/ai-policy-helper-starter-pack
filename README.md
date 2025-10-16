
  <br />
      <img src="ai-policy-helper.png">
    </a>
  <br />

  <div align="center">
    <img src="https://img.shields.io/badge/-TypeScript-black?style=for-the-badge&logoColor=white&logo=typescript&color=3178C6"/>
    <img src="https://img.shields.io/badge/-Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white" />
    <img src="https://img.shields.io/badge/Next.js-000?style=for-the-badge&logo=next.js&logoColor=white">

  </div>

<p align="center">
 
</p>

<h3 align="center">AI Policy </h3>
<p align="center"> Interview Assessment From ARRIVO 
    <br> 
</p>

---

## 📝 Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Quick Start](#quickstart)
- [Deployment](#deployment)
- [Tech Stack](#techstack)
- [Contributing](../CONTRIBUTING.md)
- [Authors](#authors)
- [Acknowledgments](#acknowledgement)
- [Reference](#reference)

## 🧐 Introduction <a name = "introduction"></a>
I was given a task by ARRIVO a startup company to build a RAG AI Policy Helper . I was given all of the codebase and folder structure my task was to polish the UI&UX , test the backend , refactor the codebase with clean and efficient code . This project will give citation 

## 🏁 Features <a name = "features"></a>

🚀 Ingestion – Ingests policy & product document

🔐 Provide Citations – Gives answers with citations title and section

🌐 Offline mode  – Can run fully offline with stub models and built-in embeddings.

🛡️ Online Mode – Can run fully online with OpenAI API and Ollama


##  🚀 Software Architecture <a name = "Software Architecture"></a>


- [Google Docs](https://docs.google.com/document/d/16Oh6VlEyygfH_A7lSQWhes0lK2TtUh-wg_njnJ9LTnQ/edit?usp=sharing) - View file to see the System Architecture 



## <a name="quick-start">🤸 Quick Start</a>

**Prerequisites**

Make sure you have the following installed on your machine:

- [Git](https://git-scm.com/)
- [Node.js](https://nodejs.org/en)
- [npm](https://www.npmjs.com/) (Node Package Manager)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Docker Desktop)



**Cloning the Repository**

```bash
git clone https://github.com/JuneshK/ai-policy-helper-starter-pack.git
cd ai-policy-helper-starter-pack
```



**Set Up Environment Variables**

Create a new file named `.env` in the root of your project and add the following content:

```env
# Backend
EMBEDDING_MODEL=local-384
LLM_PROVIDER=stub           # options: stub | openai | ollama
OPENAI_API_KEY=        # if using OpenAI, set this
OLLAMA_HOST=http://ollama:11434
VECTOR_STORE=qdrant         # qdrant | memory
COLLECTION_NAME=policy_helper
CHUNK_SIZE=700
CHUNK_OVERLAP=80

# Frontend
NEXT_PUBLIC_API_BASE=http://localhost:8000
```

Replace the placeholder values with your credentials. You can get these by signing up at: [**OpenAI API**](https://openai.com/index/openai-api/).




**Running the Project ( Method One)**

```bash
docker-compose up --build
```

Open [http://localhost:3000](http://localhost:3000) in your browser to view the Frontend <br>
Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser to view the Backend <br>
Open [http://localhost:6333](http://localhost:6333) in your browser to view the Database <br>
<br>
<small>⚠️ Note: The provided <code>docker-compose</code> setup fails due to issues with the Docker backend and connecting to the Qdrant database. As an alternative, I have implemented a <strong>manual run method</strong> for the project, which successfully starts all services and allows full functionality.</small>


<br>

**Running the Project Alternative ( Method Two )**

```bash
docker-compose up --build
```

1.Open Docker Engine<br>
 <img src="docker_engine.png"  width="500" height="500"> <br>
2. Run Each Container Manually<br>
3. Open http://localhost:3000 to see the output .


## 🔧 Running the tests <a name = "tests"></a>
Explain how to run automated tests for this system.

### Break down into unit tests
Explain what these tests test and why

```
docker compose run --rm backend pytest -q
```
<br>
<small>⚠️ Note: The provided <code>docker compose run --rm backend pytest -q</code> setup fails due to issues with the Docker backend and connecting to the Qdrant database. Hence , testing cannot be done.</small>

## 📁 Project Structure

```
ai-policy-helper/
├─ backend/
│  ├─ app/
│  │  ├─ main.py          # FastAPI app + endpoints
│  │  ├─ settings.py      # config/env
│  │  ├─ rag.py           # embeddings, vector store, retrieval, generation
│  │  ├─ models.py        # pydantic models
│  │  ├─ ingest.py        # doc loader & chunker
│  │  ├─ __init__.py
│  │  └─ tests/
│  │     ├─ conftest.py
│  │     └─ test_api.py
│  ├─ requirements.txt
│  └─ Dockerfile
├─ frontend/
│  ├─ app/
│  │  ├─ page.tsx         # chat UI
│  │  ├─ layout.tsx
│  │  └─ globals.css
│  ├─ components/
│  │  ├─ Chat.tsx
│  │  └─ AdminPanel.tsx
│  ├─ lib/api.ts
│  ├─ package.json
│  ├─ tsconfig.json
│  ├─ next.config.js
│  └─ Dockerfile
├─ data/                  # sample policy docs
├─ docker-compose.yml
├─ Makefile
└─ .env.example
```

## 🎈 Demo Video <a name="demo-video"></a>
- [Google Drive](https://drive.google.com/drive/folders/1aeYDGklDoO1dJrGWMSyuYtgyGSfx8PGa?usp=drive_link) - Link to watch the demo of this project


## ⛏️ Built Using <a name = "built_using"></a>
- [QDrant](https://qdrant.tech/) - Database
- [Next.js](https://reactnative.dev/) - Web Framework
- [TailwindCSS](https://tailwindcss.com/) - CSS Framework
- [Python](https://www.python.org/) - Backend Language
- [Typescript](https://www.typescriptlang.org/docs/) - Frontend Language


## ✍️ Authors <a name = "authors"></a>
- Myself

## 🎉 Acknowledgements <a name = "acknowledgement"></a>
- Thanks to Arrivo for this project and giving me this oppurtunity to work on this project
  
## 🎉 References<a name = "reference"></a>
- [TailwindCSS](https://tailwindcss.com/) - How to Import Tailwind on Next.js
- [HelloInterview](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery) - System Design
- [Google Docs](https://docs.google.com/document/d/1irwxfSW1CIhFNfF-QptE0EOPVLqTquxpRPxOPc1X68Q/edit?usp=sharing) - Debugging & Challenges
