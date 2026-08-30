import os
import time
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import json

from litellm import completion
# from mistralai import Mistral
from openai import OpenAI
import anthropic
# import boto3 # for aws bedrock

# from config import keys

class ModelAPIClient:
    @staticmethod
    def call_api(
        user_prompt: str,
        provider: str,
        model: str,
        max_tokens: int = 5,
        n: int = 1,
        mock: bool = False,
        system_prompt: str = None,
        temperature: float = 0.7
    ) -> Union[str, List[str]]:
        """
        Call the specified API provider to generate responses.
        
        Args:
            user_prompt: Input prompt
            provider: API provider ('openai', 'mistral', 'llama', or 'anthropic')
            model: Model name
            max_tokens: Maximum tokens in response
            n: Number of responses to generate
            mock: If True, return a random number between 0 and 100 as a string
            system_prompt: Optional system prompt to set AI behavior/role
            
        Returns:
            str or list: Generated response (or empty list if error)
        """
        # If mock is enabled, return a random number
        if mock:
            random_rating = str(np.random.randint(0, 101))  # 0 to 100 inclusive
            return f"{random_rating} BLAH"
        
        # Build messages array
        messages = []
        if system_prompt and provider in ['openai', 'mistral', 'llama']:
            # Add the max token to the system prompt to prevent errors later on from long responses
            # system_prompt += f"\nKeep your response to at most {max_tokens - 50} tokens."
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
                
        # Add small delay to avoid rate limits
        time.sleep(1)
                
        try:
            if provider == 'openai':
                if 'gpt-5' in model:
                    chat_response = completion(
                        model=f"{provider}/{model}",
                        messages=messages,
                        max_completion_tokens=max_tokens,
                        n=n,
                        reasoning_effort='low',
                    ) # gpt5 does not support temperature
                else: 
                    chat_response = completion(
                        model=f"{provider}/{model}",
                        messages=messages,
                        max_tokens=max_tokens,
                        n=n,
                        temperature=temperature
                    )
                return chat_response.choices[0].message.content
            elif provider == 'mistral':
                client = Mistral(api_key=os.environ['MISTRAL_API_KEY'])
                chat_response = client.chat.complete(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    n=n,
                    temperature=temperature
                )
                return chat_response.choices[0].message.content
            # elif provider == 'bedrock':
            #     bedrock = boto3.client('bedrock-runtime', region_name=os.environ.get('AWS_REGION', 'us-east-1'))

            #     body = {
            #         "messages": [{"role": "user", "content": user_prompt}],
            #         "max_tokens": max_tokens,
            #         "temperature": temperature,
            #         "anthropic_version": "bedrock-2023-05-31"
            #     }
            #     if system_prompt:
            #         body["system"] = system_prompt
                
            #     response = bedrock.invoke_model(
            #         modelId=model,
            #         body=json.dumps(body),
            #         accept="application/json",
            #         contentType="application/json"
            #     )
                    
            #     response_body = response["body"].read().decode()
            #     response_json = json.loads(response_body)
            #     return response_json['content'][0]['text']
            
            elif provider == 'llama':
                client = OpenAI(
                    api_key=os.environ["LLMANA_API_KEY"],
                    base_url="https://api.llama-api.com"
                )
                return client.chat.completions.create(
                    max_tokens=max_tokens,
                    model=model,
                    messages=messages,
                    temperature=temperature
                ).choices[0].message.content
            elif provider == 'anthropic':
                client = anthropic.Anthropic(
                    api_key=os.environ["ANTHROPIC_API_KEY"],
                )
                create_params = {
                    'model': model,
                    'max_tokens': max_tokens,
                    'messages': [{"role": "user", "content": user_prompt}],
                    'temperature': temperature
                }
                if system_prompt:
                    create_params['system'] = system_prompt
                # Anthropic requires streaming for requests that may run >10 minutes.
                # Streaming also works for shorter requests, so use it consistently.
                with client.messages.stream(**create_params) as stream:
                    text_chunks = [text for text in stream.text_stream]

                    if text_chunks:
                        return "".join(text_chunks)

                    # Fallback: reconstruct from final message blocks if no text events arrived.
                    final_message = stream.get_final_message()
                    return "".join(
                        block.text for block in final_message.content
                        if getattr(block, "type", None) == "text"
                    )
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        except Exception as e:
            print(f"Error calling {provider} API: {str(e)}")
            return []