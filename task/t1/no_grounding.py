import asyncio
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr
from sqlalchemy.dialects.oracle.dictionary import all_users

from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

#TODO:
# Before implementation open the `flow_diagram.png` to see the flow of app

BATCH_SYSTEM_PROMPT = """You are a user search assistant. Your task is to find users from the provided list that match the search criteria.

INSTRUCTIONS:
1. Analyze the user question to understand what attributes/characteristics are being searched for
2. Examine each user in the context and determine if they match the search criteria
3. For matching users, extract and return their complete information
4. Be inclusive - if a user partially matches or could potentially match, include them

OUTPUT FORMAT:
- If you find matching users: Return their full details exactly as provided, maintaining the original format
- If no users match: Respond with exactly "NO_MATCHES_FOUND"
- If uncertain about a match: Include the user with a note about why they might match"""

FINAL_SYSTEM_PROMPT = """You are a helpful assistant that provides comprehensive answers based on user search results.

INSTRUCTIONS:
1. Review all the search results from different user batches
2. Combine and deduplicate any matching users found across batches
3. Present the information in a clear, organized manner
4. If multiple users match, group them logically
5. If no users match, explain what was searched for and suggest alternatives"""

USER_PROMPT = """## USER DATA:
{context}

## SEARCH QUERY: 
{query}"""


class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
        self.batch_tokens = []

    def add_tokens(self, tokens: int):
        self.total_tokens += tokens
        self.batch_tokens.append(tokens)

    def get_summary(self):
        return {
            'total_tokens': self.total_tokens,
            'batch_count': len(self.batch_tokens),
            'batch_tokens': self.batch_tokens
        }

#TODO:
# 1. Create AzureChatOpenAI client
#    hint: api_version set as empty string if you gen an error that indicated that api_version cannot be None
# 2. Create TokenTracker
azure_client = AzureChatOpenAI(deployment_name="gpt-4o", azure_endpoint=DIAL_URL, api_key=SecretStr(API_KEY), api_version="")
token_tracker = TokenTracker()

def join_context(context: list[dict[str, Any]]) -> str:
    #TODO:
    # You cannot pass raw JSON with user data to LLM (" sign), collect it in just simple string or markdown.
    # You need to collect it in such way:
    # User:
    #   name: John
    #   surname: Doe
    #   ...
    joined_context = ""
    for user in context:
        joined_context += "User:\n"
        for key, value in user.items():
            joined_context += f"{key}: {value}\n"

    return joined_context


async def generate_response(system_prompt: str, user_message: str) -> str:
    print("Processing...")
    #TODO:
    # 1. Create messages array with system prompt and user message
    # 2. Generate response (use `ainvoke`, don't forget to `await` the response)
    # 3. Get usage (hint, usage can be found in response metadata (its dict) and has name 'token_usage', that is also
    #    dict and there you need to get 'total_tokens')
    # 4. Add tokens to `token_tracker`
    # 5. Print response content and `total_tokens`
    # 5. return response content
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
    response = await azure_client.ainvoke(messages)
    token_usage = response.usage_metadata
    total_tokens = token_usage.get("total_tokens", 0)
    token_tracker.add_tokens(total_tokens)
    return response.content


async def main():
    print("Query samples:")
    print(" - Do we have someone with name John that loves traveling?")

    user_question = input("> ").strip()
    if user_question:
        print("\n--- Searching user database ---")

        #TODO:
        # 1. Get all users (use UserClient)
        # 2. Split all users on batches (100 users in 1 batch). We need it since LLMs have its limited context window
        # 3. Prepare tasks for async run of response generation for users batches:
        #       - create array tasks
        #       - iterate through `user_batches` and call `generate_response` with these params:
        #           - BATCH_SYSTEM_PROMPT (system prompt)
        #           - User prompt, you need to format USER_PROMPT with context from user batch and user question
        # 4. Run task asynchronously, use method `gather` form `asyncio`
        # 5. Filter results on 'NO_MATCHES_FOUND' (see instructions for BATCH_SYSTEM_PROMPT)
        # 5. If results after filtration are present:
        #       - combine filtered results with "\n\n" spliterator
        #       - generate response with such params:
        #           - FINAL_SYSTEM_PROMPT (system prompt)
        #           - User prompt: you need to make augmentation of retrieved result and user question
        # 6. Otherwise prin the info that `No users found matching`
        # 7. In the end print info about usage, you will be impressed of how many tokens you have used. (imagine if we have 10k or 100k users 😅)
        client_manger = UserClient()
        all_users = client_manger.get_all_users()
        user_batches = [all_users[i:i + 100] for i in range(0, len(all_users), 100)]
        tasks = []
        for batch in user_batches:
            context = join_context(batch)
            user_prompt = USER_PROMPT.format(context=context, query=user_question)
            tasks.append(generate_response(BATCH_SYSTEM_PROMPT, user_prompt))
        batch_results = await asyncio.gather(*tasks)
        filtered_results = [result for result in batch_results if result != "NO_MATCHES_FOUND"]
        if filtered_results:
            combined_results = "\n\n".join(filtered_results)
            final_user_prompt = f'## SEARCH RESULTS:\n{combined_results}\n\n## ORIGINAL QUERY:\n{user_question}'
            final_response = await generate_response(FINAL_SYSTEM_PROMPT, final_user_prompt)
            print("\n--- Final Search Results ---")
            print(final_response)
        else:
            print("No users found matching the criteria.")

        usage_summary = token_tracker.get_summary()
        print("\n--- Usage Summary ---")
        print(f"Total Tokens Used: {usage_summary['total_tokens']}")


if __name__ == "__main__":
    asyncio.run(main())


# The problems with No Grounding approach are:
#   - If we load whole users as context in one request to LLM we will hit context window
#   - Huge token usage == Higher price per request
#   - Added + one chain in flow where original user data can be changed by LLM (before final generation)
# User Question -> Get all users -> ‼️parallel search of possible candidates‼️ -> probably changed original context -> final generation