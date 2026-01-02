import asyncio
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr, BaseModel, Field, RootModel
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

#TODO: Info about app:
# HOBBIES SEARCHING WIZARD
# Searches users by hobbies and provides their full info in JSON format:
#   Input: `I need people who love to go to mountains`
#   Output:
#     ```json
#       "rock climbing": [{full user info JSON},...],
#       "hiking": [{full user info JSON},...],
#       "camping": [{full user info JSON},...]
#     ```
# ---
# 1. Since we are searching hobbies that persist in `about_me` section - we need to embed only user `id` and `about_me`!
#    It will allow us to reduce context window significantly.
# 2. Pay attention that every 5 minutes in User Service will be added new users and some will be deleted. We will at the
#    'cold start' add all users for current moment to vectorstor and with each user request we will update vectorstor on
#    the retrieval step, we will remove deleted users and add new - it will also resolve the issue with consistency
#    within this 2 services and will reduce costs (we don't need on each user request load vectorstor from scratch and pay for it).
# 3. We ask LLM make NEE (Named Entity Extraction) https://cloud.google.com/discover/what-is-entity-extraction?hl=en
#    and provide response in format:
#    {
#       "{hobby}": [{user_id}, 2, 4, 100...]
#    }
#    It allows us to save significant money on generation, reduce time on generation and eliminate possible
#    hallucinations (corrupted personal info or removed some parts of PII (Personal Identifiable Information)). After
#    generation we also need to make output grounding (fetch full info about user and in the same time check that all
#    presented IDs are correct).
# 4. In response we expect JSON with grouped users by their hobbies.
# ---
# This sample is based on the real solution where one Service provides our Wizard with user request, we fetch all
# required data and then returned back to 1st Service response in JSON format.
# ---
# Useful links:
# Chroma DB: https://docs.langchain.com/oss/python/integrations/vectorstores/index#chroma
# Document#id: https://docs.langchain.com/oss/python/langchain/knowledge-base#1-documents-and-document-loaders
# Chroma DB, async add documents: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.aadd_documents
# Chroma DB, get all records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.get
# Chroma DB, delete records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.delete
# ---
# TASK:
# Implement such application as described on the `flow.png` with adaptive vector based grounding and 'lite' version of
# output grounding (verification that such user exist and fetch full user info)



SYSTEM_PROMPT = """You are a hobby searching wizard that helps to find users based on their hobbies. 
## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Response Format:
{format_instructions}

The response MUST be a JSON object where:
- each key is a hobby name (string)
- each value is an array of user_id (integers)

## Example:
{{
  "chess": [100, 204],
  "football": [300]
}}
"""
USER_PROMPT = """## RAG CONTEXT: {context}
## USER QUESTION: {query}"""

llm_client = AzureChatOpenAI(deployment_name="gpt-4o", azure_endpoint=DIAL_URL, api_key=SecretStr(API_KEY), api_version="")
user_client = UserClient()

class HobbyUsersResponse(RootModel):
    root: dict[str, list[int]]


def create_vector_store() -> Chroma:
    """Create and return a Chroma vector store ."""
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-3-small-1",
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        api_version="",
        dimensions=384,
    )

    vector_store = Chroma(embedding_function=embeddings, collection_name="users_hobbies_collection", persist_directory="./chroma_db")

    return vector_store


async def load_vector_store(vector_store: Chroma, batch_size: int = 100):
    """Load all users from User Service and add their id and about_me to vector store."""
    all_users = user_client.get_all_users()
    batches = [all_users[i:i + batch_size] for i in range(0, len(all_users), batch_size)]
    for batch in batches:
        documents = [
            Document(page_content=user["about_me"], metadata={"id": user["id"]}, id=str(user["id"]))
            for user in batch
        ]
        await vector_store.aadd_documents(documents)

    print(f"Loaded {len(all_users)} users into vector store.")

async def update_local_vector_store(vector_store: Chroma):
    """Update vector store by removing deleted users and adding new users."""
    all_users = user_client.get_all_users()
    store_dump = vector_store.get()
    existing_ids = set(map(str, store_dump["ids"]))

    current_ids = {str(user["id"]) for user in all_users}

    # Identify deleted users
    deleted_ids = existing_ids - current_ids
    if deleted_ids:
        await vector_store.adelete(ids=list(deleted_ids))
        print(f"Removed {len(deleted_ids)} deleted users from vector store.")

    # Identify new users
    new_users = [user for user in all_users if str(user["id"]) not in existing_ids]
    new_documents = [
        Document(page_content=user["about_me"], metadata={"id": user["id"]}, id=str(user["id"]))
        for user in new_users
    ]
    if new_documents:
        await vector_store.aadd_documents(new_documents)
        print(f"Added {len(new_documents)} new users to vector store.")


async def retrieve_hobby_users(user_question: str, vector_store: Chroma, k=10, score: float = 0.1) -> str:
    await update_local_vector_store(vector_store)

    relevant_docs = await vector_store.asimilarity_search_with_relevance_scores(user_question, k, score_threshold=score)
    context_parts = []
    for doc, relevance_score in relevant_docs:
        user_id = doc.metadata.get("id")
        context_parts.append(
            f"USER_ID: {user_id}\nABOUT_ME:\n{doc.page_content}"
        )

        print(f"Score: {relevance_score}\nContent:\n{doc.page_content}\n")

    return "\n\n---\n\n".join(context_parts)


def augment_prompt(user_question: str, context: str) -> str:
    augmented_prompt = USER_PROMPT.format(context=context, query=user_question)
    return augmented_prompt


async def generate_answer(augmented_prompt: str) -> str:
    parser = PydanticOutputParser(pydantic_object=HobbyUsersResponse)
    messages = [
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        HumanMessage(content=augmented_prompt)
    ]
    prompt = ChatPromptTemplate.from_messages(messages=messages).partial(format_instructions=parser.get_format_instructions())
    hobby_users_response = await (prompt | llm_client | parser).ainvoke({})
    return hobby_users_response.model_dump_json()


async def process_llm_response(llm_response: str) -> dict[str, list[dict[str, Any]]]:
    """Process LLM response to fetch full user info and group by hobbies."""
    hobby_users = {}
    parsed_response = HobbyUsersResponse.model_validate_json(llm_response)

    for hobby, user_ids in parsed_response.root.items():
        tasks = [user_client.get_user(user_id) for user_id in user_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        users_info = []
        for user_id, result in zip(user_ids, results):
            if isinstance(result, Exception):
                print(f"Error fetching user {user_id}: {result}")
            else:
                users_info.append(result)

        hobby_users[hobby] = users_info

    return hobby_users


async def main():
    vector_store = create_vector_store()
    store_dump = vector_store.get()
    if not store_dump["ids"]:
        await load_vector_store(vector_store)

    while True:
        user_question = input("Enter your hobby search question (or 'exit' to quit): ")
        if user_question.lower() == 'exit':
            break

        context = await retrieve_hobby_users(user_question, vector_store)
        augmented_prompt = augment_prompt(user_question, context)
        llm_response = await generate_answer(augmented_prompt)
        hobby_users = await process_llm_response(llm_response)

        print("Hobby Users Response:")
        print(hobby_users)

if __name__ == "__main__":
    asyncio.run(main())