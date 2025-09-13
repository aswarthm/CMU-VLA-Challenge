from flask import Flask, request, jsonify
from google import genai
from google.genai.types import Tool, FunctionDeclaration, GenerateContentConfig, ToolConfig, FunctionCallingConfig, Content, Part, ThinkingConfig
from pydantic import BaseModel, Field
from typing import Literal, Union
from enum import Enum
import json

from pydanticModels import *


app = Flask(__name__)
genai_client = genai.Client()

conversation_history = []
thinking_budget = -1 # 1024*32


@app.route('/reset_conversation', methods=['POST'])
def reset_conversation():
    conversation_history.clear()
    return str(len(conversation_history)) + "reset done"


# @app.route('/determine_task_type', methods=['POST'])
# def determine_task_type():
#     question = ChallengeQuestion.model_validate_json(
#         request.data).challenge_question
#     prompt = f"""
#                     Classify the following question
#                     {question}
#               """
#     response = genai_client.models.generate_content(
#         model="gemini-2.5-pro",
#         contents=[prompt],
#         config=GenerateContentConfig(
#             response_mime_type="application/json",
#             response_schema=ClassifyChallenge
#         )
#     )
#     return str(response)


@app.route('/fn_call', methods=['POST'])
def fn_call():
    global conversation_history
    print("Processing new request")
    req = VLMRequest.model_validate_json(request.data)
    SYSTEM_PROMPT = """
                        
                        You are the central reasoning core and strategic planner for an autonomous robot. Your purpose is to solve the user's mission by relying on your powerful visual understanding of the environment.

                        OPERATIONAL PROCEDURE & CONSTRAINTS:

                        1.  Analyze: Your first step is to analyze the `challenge_question` and the attached camera `image`.
                        2. FOR NUMERICAL COUNTING TASKS:
                            Your instantaneous camera view is incomplete and **cannot** be trusted for an accurate count. Objects may be hidden or outside your current field of view.
                            To get a reliable count, you **MUST** first build a complete memory of all target objects by exploring the environment.
                            To do this, you **MUST** call the `describe_point_fn` tool one or more times to move to different viewpoints until you are confident you have scanned all accessible areas.
                            Only after you have finished this comprehensive visual survey are you allowed to call the `submit_final_count` tool. 
                        3.  FOR OBJECT REFERENCE & NAVIGATION TASKS:
                            To identify a specific object, follow a path, or navigate to a location, your primary tool is `describe_point_fn`.
                            You **MUST** first analyze the image to identify the most likely visual target that satisfies the user's request.
                            After identifying the target, you **MUST** call the `describe_point_fn` tool, providing the `[y, x]` pixel coordinates corresponding to the center of the base of that target.
                        4.  FOR INSTRUCTION FOLLOWING TASKS:
                            Your goal is to navigate along a path defined by a sequence of landmarks.
                            You must visit each landmark in the correct order.
                            To visit a landmark, you must first use the `describe_point_fn` tool to move the robot to its location.
                            After the navigation is complete, you will receive a new world state. You can only consider a landmark "visited" in your plan if the robot's current position is in **close proximity** to that landmark (e.g., with a green box around it).
                            **Just seeing a landmark from a distance is not sufficient to mark it as visited.** Proceed to the next landmark only after confirming close proximity to the current one.
                        5.  MANDATORY ACTION VERIFICATION:
                            After you call a navigation tool like `describe_point_fn`, the system will execute the move and then provide you with a new camera image from the robot's resulting state.
                            Your next step MUST be to analyze this new image to get visual confirmation that the action was successful and the robot is in the correct location before proceeding with the plan.
                        6.  Communication Protocol: You MUST respond ONLY by calling one of the provided functions. Do not respond with conversational text.

                        All mission goals are achievable. If you encounter a failure, an obstacle, or a dead end in your plan, you are not to give up. Your protocol is to re-analyze the situation, formulate a new hypothesis, and attempt a different sequence of tool calls to overcome the obstacle and complete the mission.
                    """

    current_state_prompt = f"""

                                **MISSION GOAL:**
                                {req.challenge_question}

                                **CURRENT VIEW**
                                {req.current_view}
                                
                            """

                                # **CURRENT PLAN STATE:**
                                # - Tasks: {req.tasks if req.tasks else "No plan exists yet. Your first job is to create one."}
    print(len(conversation_history))
    tool_config = ToolConfig(
        function_calling_config=FunctionCallingConfig(
            mode="ANY"
        )
    )
    all_tools = Tool(
        function_declarations=[
            # set_plan_fn,
            # describe_point_fn,
            navigate_to_point_fn,
            verify_object_exists_fn,
            # request_filtered_view_fn,
            # get_visual_confirmation_fn,
            submit_final_count_fn,
            submit_final_object_reference_fn,
            finish_instruction_following_fn,
        ]
    )
    config = GenerateContentConfig(
        thinking_config=ThinkingConfig(thinking_budget=thinking_budget),
        tools=[all_tools],
        tool_config=tool_config,
        system_instruction=SYSTEM_PROMPT,
        temperature=0.5
    )
    image_part = Part(
        inline_data={       
            "mime_type": "image/png",  # or "image/jpeg" etc.
            "data": req.image
        }
    )
    response = genai_client.models.generate_content(
        model="gemini-2.5-pro",
        contents=conversation_history + [Content(
            role="user", parts=[Part(text=current_state_prompt), image_part]
        )],
        config=config
    )
    contents = [
        Content(
            role="user", parts=[Part(text=current_state_prompt)]
        )
    ]
    conversation_history.append(contents[0])
    conversation_history.append(response.candidates[0].content)
    # conversation_history = conversation_history[-16:]
    print("Finished processing")
    return jsonify({
        "args": response.candidates[0].content.parts[0].function_call.args,
        "name": response.candidates[0].content.parts[0].function_call.name
    })


@app.route('/get_pixel', methods=['POST'])
def get_pixel():
    print("Processing get pixel request")
    req = request.json
    prompt = f'''You are a visual grounding system for a robot. Your task is to find potential navigation targets in the provided image that match the user's request and return the top 5 possibilities.

                    **TARGET DESCRIPTION:**
                    {req.get('description')}

                    **INSTRUCTIONS:**
                    1.  Analyze the target description and the attached image.
                    2.  Identify all potential objects that could match the description.
                    3.  For each potential object, select a point at the **center of its base**.
                    4.  Your final action MUST be to call the `navigate_to_point_fn` function.
                    5.  You MUST return a list of 5 points, sorted with the most likely target first.
                '''
    tool_config = ToolConfig(
        function_calling_config=FunctionCallingConfig(
            mode="ANY"
        )
    )
    config = GenerateContentConfig(
        thinking_config=ThinkingConfig(thinking_budget=thinking_budget),
        tools=[Tool(function_declarations=[navigate_to_point_fn])],
        tool_config=tool_config,
        temperature=0.5

    )
    image_part = Part(
        inline_data={
            "mime_type": "image/png",  # or "image/jpeg" etc.
            "data": req.get('image')
        }
    )
    contents = [
        Content(
            role="user", parts=[Part(text=prompt), image_part]
        )
    ]
    response = genai_client.models.generate_content(
        model="gemini-2.5-pro",
        contents=contents,
        config=config
    )
    return jsonify({
        "args": response.candidates[0].content.parts[0].function_call.args,
        "name": response.candidates[0].content.parts[0].function_call.name
    })


@app.route('/add_tool_response', methods=['POST'])
def add_tool_response():
    """
    Receives the result of a tool call (an observation), adds it to history,
    and returns Gemini's NEXT tool call.
    """
    observation = request.json

    
    img = None
    parts = [Part(text=json.dumps(observation))]
    if "img" in observation:
        img = observation.pop("img")
        image_part = Part(
            inline_data={
                "mime_type": "image/png",  # or "image/jpeg" etc.
                "data": img
            }
        )
        parts.append(image_part)

    tool_response_part = [
        Content(
            role="user", parts=parts
        )
    ]
    conversation_history.append(tool_response_part[0])

    tool_config = ToolConfig(
        function_calling_config=FunctionCallingConfig(
            mode="ANY"
        )
    )
    all_tools = Tool(
        function_declarations=[
            # set_plan_fn,
            # describe_point_fn,
            navigate_to_point_fn,
            verify_object_exists_fn,
            # get_visual_confirmation_fn,
            # request_filtered_view_fn,
            submit_final_count_fn,
            submit_final_object_reference_fn,
            finish_instruction_following_fn,
        ]
    )
    config = GenerateContentConfig(
        thinking_config=ThinkingConfig(thinking_budget=thinking_budget),
        tools=[all_tools],
        tool_config=tool_config,
        temperature=0.5
    )
    response = genai_client.models.generate_content(
        model="gemini-2.5-pro",
        contents=conversation_history,
        config=config
    )
    conversation_history.append(response.candidates[0].content)
    print("Finished processing tool response")
    return jsonify({
        "args": response.candidates[0].content.parts[0].function_call.args,
        "name": response.candidates[0].content.parts[0].function_call.name
    })

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
