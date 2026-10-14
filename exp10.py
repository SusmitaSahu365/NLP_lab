# ============================================================
# EXPERIMENT 10
# Exploring and Implementing Large Language Models
# using LangChain and LangGraph
# Streamlit Interface
# ============================================================

import streamlit as st
from typing import TypedDict

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LLM Explorer",
    page_icon="",
    layout="centered"
)


# ============================================================
# PAGE TITLE
# ============================================================

st.title("LLM Topic Explorer")
st.caption("Exploring Large Language Models using LangChain + LangGraph")


st.divider()


# ============================================================
# API KEY INPUT
# ============================================================

st.subheader(" Groq API Configuration")

api_key = st.text_input(
    "Enter your Groq API Key",
    type="password",
    placeholder="gsk_..."
)

st.caption(
    "Your API key is used only for this session and is not displayed."
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "openai/gpt-oss-20b"


# ============================================================
# LANGGRAPH STATE
# ============================================================

class GraphState(TypedDict):
    topic: str
    analysis: str
    answer: str
    final_answer: str


# ============================================================
# CREATE LANGGRAPH APPLICATION
# ============================================================

def create_graph(api_key):

    # --------------------------------------------------------
    # Initialize Groq LLM
    # --------------------------------------------------------

    llm = ChatGroq(
        api_key=api_key,
        model=MODEL_NAME,
        temperature=0.7
    )

    # --------------------------------------------------------
    # LANGCHAIN PROMPTS
    # --------------------------------------------------------

    analysis_prompt = ChatPromptTemplate.from_template(
        """
You are an NLP expert.

Analyze the following topic:

Topic: {topic}

Provide:
1. Definition
2. Main concepts
3. Applications
4. Importance

Keep the explanation suitable for a college student.
"""
    )

    answer_prompt = ChatPromptTemplate.from_template(
        """
You are a helpful AI tutor.

Using the analysis below, generate a clear educational
answer about the topic.

Topic:
{topic}

Analysis:
{analysis}

Explain the topic in simple language and include examples
where appropriate.
"""
    )

    improve_prompt = ChatPromptTemplate.from_template(
        """
You are an editor.

Improve the following answer so that it is:

- clear
- concise
- technically correct
- easy for a college student to understand

Topic:
{topic}

Answer:
{answer}

Return only the improved answer.
"""
    )

    # --------------------------------------------------------
    # LANGCHAIN CHAINS
    # --------------------------------------------------------

    analysis_chain = analysis_prompt | llm
    answer_chain = answer_prompt | llm
    improve_chain = improve_prompt | llm

    # --------------------------------------------------------
    # LANGGRAPH NODES
    # --------------------------------------------------------

    def analyze_topic(state: GraphState):

        response = analysis_chain.invoke({
            "topic": state["topic"]
        })

        return {
            "analysis": response.content
        }

    def generate_answer(state: GraphState):

        response = answer_chain.invoke({
            "topic": state["topic"],
            "analysis": state["analysis"]
        })

        return {
            "answer": response.content
        }

    def improve_answer(state: GraphState):

        response = improve_chain.invoke({
            "topic": state["topic"],
            "answer": state["answer"]
        })

        return {
            "final_answer": response.content
        }

    # --------------------------------------------------------
    # BUILD LANGGRAPH
    # --------------------------------------------------------

    workflow = StateGraph(GraphState)

    workflow.add_node("analyze", analyze_topic)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("improve", improve_answer)

    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", "generate")
    workflow.add_edge("generate", "improve")
    workflow.add_edge("improve", END)

    return workflow.compile()


# ============================================================
# TOPIC INPUT
# ============================================================

st.subheader(" Enter Topic")

topic = st.text_input(
    "What would you like the LLM to explain?",
    placeholder="Example: Neural Networks"
)


# ============================================================
# GENERATE BUTTON
# ============================================================

generate_button = st.button(
    " Generate Answer",
    use_container_width=True
)


# ============================================================
# EXECUTE LANGGRAPH
# ============================================================

if generate_button:

    # --------------------------------------------------------
    # Validate API Key
    # --------------------------------------------------------

    if not api_key:

        st.error(
            "Please enter your Groq API key."
        )

        st.stop()

    # --------------------------------------------------------
    # Validate Topic
    # --------------------------------------------------------

    if not topic.strip():

        st.warning(
            "Please enter a topic."
        )

        st.stop()

    # --------------------------------------------------------
    # Create Graph
    # --------------------------------------------------------

    try:

        graph = create_graph(api_key)

        # ----------------------------------------------------
        # Initial State
        # ----------------------------------------------------

        initial_state = {
            "topic": topic,
            "analysis": "",
            "answer": "",
            "final_answer": ""
        }

        # ----------------------------------------------------
        # Progress Display
        # ----------------------------------------------------

        with st.status(
            "Running LangGraph workflow...",
            expanded=True
        ) as status:

            st.write(" Analyzing topic...")

            result = graph.invoke(initial_state)

            st.write(" Generating answer...")

            st.write(" Improving final answer...")

            status.update(
                label="LangGraph workflow completed!",
                state="complete"
            )

        # ----------------------------------------------------
        # DISPLAY ANALYSIS
        # ----------------------------------------------------

        st.divider()

        st.subheader("Topic Analysis")

        st.write(result["analysis"])

        # ----------------------------------------------------
        # DISPLAY GENERATED ANSWER
        # ----------------------------------------------------

        st.divider()

        st.subheader("Generated Answer")

        st.write(result["answer"])

        # ----------------------------------------------------
        # DISPLAY FINAL ANSWER
        # ----------------------------------------------------

        st.divider()

        st.subheader("✨ Final Improved Answer")

        st.success(result["final_answer"])

        # ----------------------------------------------------
        # WORKFLOW
        # ----------------------------------------------------

        st.divider()

        st.subheader("🔗 LangGraph Workflow")

        st.code(
            """
START
  ↓
Analyze Topic
  ↓
Generate Answer
  ↓
Improve Answer
  ↓
END
""",
            language="text"
        )

        

    # --------------------------------------------------------
    # ERROR HANDLING
    # --------------------------------------------------------

    except Exception as e:

        st.error("An error occurred while running the LLM.")

        st.code(str(e))

        st.info(
            "If the error says 'Invalid API Key', create a "
            "new Groq API key and enter it above."
        )


# ============================================================
# INFORMATION SECTION
# ============================================================

st.divider()

with st.expander("About this Experiment"):

    st.write(
        """
This experiment demonstrates the implementation of a Large
Language Model application using LangChain and LangGraph.

LangChain is used for:

• Prompt templates
• LLM interaction
• Sequential processing

LangGraph is used for:

• Maintaining application state
• Creating workflow nodes
• Controlling the execution flow

The application uses a three-step workflow:

1. Analyze the topic
2. Generate an educational answer
3. Improve the generated answer
"""
    )