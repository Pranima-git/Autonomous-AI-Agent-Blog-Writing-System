# 🧠 AI Agent Blog Writing Project (CrewAI + LLaMA3)

This project demonstrates the use of autonomous AI agents to collaboratively plan and write a blog post. It uses the [CrewAI](https://docs.crewai.com/) framework along with the LLaMA3 language model (via Groq) to simulate agent collaboration for content generation.

## 📌 Project Goal
To show how multiple AI agents can work together in sequence to complete a task — specifically:
- One agent creates a detailed outline.
- Another agent writes a full blog based on that outline.

## ⚙️ Technologies Used
- Python
- [CrewAI](https://docs.crewai.com/)
- [LangChain Groq](https://python.langchain.com/docs/integrations/llms/groq/)
- LLaMA3-8B model via Groq
- .env for API key handling
- VS Code

## 💡 How It Works
1. The user defines a topic — e.g., "Top 5 AI Tools for Students in 2025".
2. The **Content Planner Agent** creates a structured outline.
3. The **Blog Writer Agent** expands that outline into a full blog post.
4. The result is printed in the terminal.

## 🧪 How to Test It
- Run the script in VS Code.
- Check the terminal output.
- If the blog is well-structured and matches the topic, the agents are working correctly.

## 📄 Sample Output
> 📄 FINAL BLOG OUTPUT:
>
> Top 5 AI Tools for Students in 2025: Revolutionizing Education**

As we step into the new decade, Artificial Intelligence (AI) is transforming various aspects of our lives, including education. AI tools are becoming increasingly popular among students, teachers, and educators, revolutionizing the way we learn and interact with information. In this article, we'll explore the top 5 AI tools for students in 2025, showcasing their potential to enhance learning outcomes, streamline academic tasks, and prepare students for the future.

**1. AI-Powered Learning Platforms: The Future of Education**

AI-powered learning platforms are designed to personalize learning experiences, making education more effective and engaging. These platforms use machine learning algorithms to analyze student data, identifying strengths, weaknesses, and learning patterns. Some of the top AI-powered learning platforms include:

* **Knewton**: This platform uses AI to create personalized learning plans, adapting to individual students' needs and abilities. Knewton's AI engine analyzes student performance data, providing real-time feedback and recommendations.
* **DreamBox**: DreamBox is an AI-powered math learning platform that uses game-like interactions to engage students. The platform's AI engine adapts to individual students' skills, providing targeted instruction and feedback.
