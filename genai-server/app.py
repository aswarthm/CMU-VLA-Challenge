from flask import Flask, request
from google import genai
from google.genai.types import Tool, FunctionDeclaration, GenerateContentConfig, ToolConfig, FunctionCallingConfig, Content, Part
from pydantic import BaseModel, Field
from typing import Literal, Union
from enum import Enum

from pydanticModels import ChallengeQuestion, VLMRequest, ClassifyChallenge, NavigateToObject, SubmitFinalCount, SubmitFinalObjectReference, FinishInstructionFollowing, Tasks, all_tools


app = Flask(__name__)
genai_client = genai.Client()

conversation_history = []


@app.route('/determine_task_type', methods=['POST'])
def determine_task_type():
    question = ChallengeQuestion.model_validate_json(
        request.data).challenge_question
    prompt = f"""
                    Classify the following question
                    {question}
              """
    response = genai_client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[prompt],
        config=GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ClassifyChallenge
        )
    )
    return str(response)


@app.route('/fn_call', methods=['POST'])
def fn_call():
    req = VLMRequest.model_validate_json(
        request.data)
    SYSTEM_PROMPT = """
                        You are the central reasoning core and strategic planner for an autonomous robot.
                        Your sole purpose is to solve the user's mission by creating and executing a plan using the provided tools.

                        **INSTRUCTIONS & CONSTRAINTS:**
                        1.  Analyze the mission goal and the current world state provided by the user on each turn.
                        3.  a plan already exists, identify the next pending subtask and call the appropriate tool (`find_objects`, `Maps_to_object`, etc.) to execute it.
                        4.  You MUST respond ONLY by calling one of the provided functions. Do not respond with conversational text.
                        5.  You MUST use `find_objects` to uniquely identify any target object before you attempt to navigate to it or submit an answer about it. If ambiguity exists, you MUST ask for clarification.
                    """
    
    current_state_prompt = f"""

                                **MISSION GOAL:**
                                {req.challenge_question}

                                **CURRENT WORLD STATE:**
                                - Currently Visible Objects: {req.current_view}
                                - Current Position: x: 0.02699209190905094
                                                    y: -0.8893573880195618
                                                    z: 0.75

    orientation:
                                **CURRENT PLAN STATE:**
                                - Tasks: {req.tasks if req.tasks else "No plan exists yet. Your first job is to create one."}
                                
                            """
    print(len(conversation_history))
    tool_config = ToolConfig(
        function_calling_config=FunctionCallingConfig(
            mode="ANY"
        )
    )
    config = GenerateContentConfig(
        tools=[all_tools],
        tool_config=tool_config, 
        system_instruction= SYSTEM_PROMPT
    )
    image_part = Part(
        inline_data={
            "mime_type": "image/png",  # or "image/jpeg" etc.
            "data": req.image
        }
    )
    contents = [
        Content(
            role="user", parts=[Part(text=current_state_prompt), image_part]
        )
    ]
    conversation_history.append(contents[0])
    response = genai_client.models.generate_content(
        model="gemini-2.5-pro",
        contents=conversation_history,
        config=config
    )
    conversation_history.append(response.candidates[0].content)
    return str(response.candidates[0].content)


# @app.route('/generate_plan', methods=['POST'])
# def generate_plan():
#     req = VLMRequest.model_validate_json(
#         request.data)
#     prompt = f"""
#                     Classify the following question
#                     {question}
#               """
#     response = genai_client.models.generate_content(
#         model="gemini-2.5-pro",
#         contents=[prompt],
#         config=GenerateContentConfig(
#             response_mime_type="application/json",
#             response_schema=Tasks
#         )
#     )
#     return str(response)


# @app.route('/vlm_loop', methods=['POST'])
# def vlm_loop():
#     req = VLMRequest.model_validate_json(
#         request.data)

#     prompt = f"""
#                     Classify the following question
#                     {question}
#               """

#     response = genai_client.models.generate_content(
#         model="gemini-2.5-pro",
#         contents=[prompt],
#         config=GenerateContentConfig(
#             response_mime_type="application/json",
#             response_schema=Union[NavigateToObject, SubmitFinalCount,
#                                   SubmitFinalObjectReference, FinishInstructionFollowing]
#         )
#     )
#     return str(response)


if __name__ == '__main__':
    app.run(debug=True)
