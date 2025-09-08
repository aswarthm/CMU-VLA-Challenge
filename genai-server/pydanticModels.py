from flask import Flask, request
from google import genai
from google.genai.types import Tool, FunctionDeclaration, GenerateContentConfig, ToolConfig, FunctionCallingConfig, Content, Part
from pydantic import BaseModel, Field
from typing import Literal, Union, Any, List, Dict
from enum import Enum


class ChallengeQuestion(BaseModel):
    """
    Models the initial, unmodified query received from the challenge system.
    """
    challenge_question: str = Field(
        ..., description="The raw, natural language query or instruction from the user that the robot must solve.")


class Subtask(BaseModel):
    """
    Models a single, discrete step within a larger plan that the AI is executing
    to solve the main challenge.
    """
    description: str = Field(...,
                             description="A concise, human-readable summary of the specific action or goal for this step, e.g., 'Find all objects with the label chair'.")
    completed: bool = Field(...,
                            description="The status of this subtask. 'False' means it is pending or in progress. 'True' means it has been successfully finished.")


class Tasks(BaseModel):
    tasks: List[Subtask] = Field(..., description="")


class WorldObject(BaseModel):
    """Describes a single object currently within the robot's field of view."""
    id: int = Field(..., description="The unique identifier for this object, e.g., 'chair_5'.")
    type: str = Field(...,
                      description="The semantic label of the object, e.g., 'chair'.")
    pose: List[float] = Field(
        ..., description="The [X, Y, Z] coordinate of the object's center in the map frame.")


class VLMRequest(BaseModel):
    """
    Represents the complete state-of-the-world payload sent from the ROS system
    to the VLM server for a single reasoning step.
    """
    challenge_question: str = Field(
        ..., description="The raw, natural language query or instruction from the user that the robot must solve.")
    tasks: List[Subtask] = Field(
        ..., description="The AI's current step-by-step plan. This shows what has been completed, what is in progress, and what is pending.")
    current_view: List[WorldObject] = Field(
        ..., description="The robot's immediate perception, containing a list of objects currently visible in its camera feed.")
    # all_objects_discovered: Dict[str, Any] = Field(
    #     ..., description="The robot's long-term memory, containing a summary of all unique objects found since the task began.")
    image: str = Field(
        ..., description="A required base64 encoded string of the robot's current camera view, providing visual context for the reasoning step.")


class ClassifyChallenge(BaseModel):
    """
    Select the single best category that describes the user's request.
    """
    task_type: Literal[
        "Numerical",
        "Object Reference",
        "Instruction Following"
    ] = Field(..., description=(
        "Numerical: The user is asking 'how many' of something. The final answer is an integer count. "
        "Object Reference: The user wants to find a single, specific object based on its attributes or spatial relationships. "
        "Instruction Following: The user is providing a sequence of commands to define a path for the robot to take."
    ))


# class NavigateToObject(BaseModel):
#     """
#     Issues a command to the robot's navigation system to move to a safe position
#     near a single, specific, known object. Only call this after the target object
#     has been uniquely identified and movement is required.
#     """
#     reasoning: str = Field(..., 
#                            description="Explain why navigating to this specific object is the correct next step in the plan.")
#     target_id: str = Field(..., 
#                            description="The unique and unambiguous ID of the target object. This ID MUST be one of the IDs listed in the 'Currently Visible Objects' section of the CURRENT WORLD STATE.")

# navigate_to_object_fn = FunctionDeclaration(
#     name="navigate_to_object",
#     description="Issues a command to the robot's navigation system to move near a specific, known object.",
#     parameters=NavigateToObject.model_json_schema()
# )

class NavigateToObject(BaseModel):
    """
    Selects a specific point in the camera's visual field and commands the robot
    to navigate to the corresponding real-world location on the floor. Use this to
    specify a direct navigation target from the image.
    """
    reasoning: str = Field(...,
                           description="A clear, step-by-step explanation for why this specific visual point was chosen as the navigation target.")
    point_x: float = Field(...,
                         description="The horizontal pixel coordinate of the target point, normalized to a 0-1000 scale, where 0 is the far left of the image.")
    point_y: float = Field(...,
                         description="The vertical pixel coordinate of the target point, normalized to a 0-1000 scale, where 0 is the top of the image.")

navigate_to_object_fn = FunctionDeclaration(
    name="navigate_to_point",
    description="Selects a specific point in the camera's visual field and commands the robot to navigate to the corresponding real-world location on the floor.",
    parameters=NavigateToObject.model_json_schema()
)


class SubmitFinalCount(BaseModel):
    """
    Submits the final integer answer for a 'numerical' type question. Use this
    tool only when you have a definitive count of the requested objects.
    """
    reasoning: str = Field(..., description="Explain how you arrived at this final count and confirm that the task is complete.")
    answer: int = Field(..., description="The final integer count, e.g., 4.")


submit_final_count_fn = FunctionDeclaration(
    name="submit_final_count",
    description="Submits the final integer answer for a 'numerical' type question.",
    parameters=SubmitFinalCount.model_json_schema()
)


class SubmitFinalObjectReference(BaseModel):
    """
    Submits the final object ID for an 'object_reference' type question. Use this
    tool only when you have uniquely identified the single object the user is referring to.
    """
    reasoning: str = Field(..., description="Explain why this specific object is the correct final answer and confirm the task is complete.")
    answer: str = Field(
        ..., description="The unique string ID of the single target object, e.g., 'chair_5'.")


submit_final_object_reference_fn = FunctionDeclaration(
    name="submit_final_object_reference",
    description="Submits the final object ID for an 'object_reference' type question.",
    parameters=SubmitFinalObjectReference.model_json_schema()
)


class FinishInstructionFollowing(BaseModel):
    """
    Submits the completion signal for an 'instruction_following' type task. Use this
    tool only after the robot has successfully completed the entire requested navigation path.
    """
    reasoning: str = Field(..., description="Confirm that all steps of the navigation instruction have been completed successfully.")
    answer: str = Field(
        "Path complete.", description="A confirmation message indicating the navigation task is finished.")


finish_instruction_following_fn = FunctionDeclaration(
    name="finish_instruction_following",
    description="Submits the completion signal for an 'instruction_following' type task.",
    parameters=FinishInstructionFollowing.model_json_schema()
)


class SetPlan(BaseModel):
    """
    Generates and sets the initial step-by-step plan for solving the challenge.
    This MUST be the first tool called when no task plan exists.
    """
    reasoning: str = Field(
        ..., description="A step-by-step explanation of why this plan is correct and how it will solve the user's request.")
    tasks: List[str] = Field(
        ..., description="A complete list of all subtasks required to solve the user's request.")


set_plan_fn = FunctionDeclaration(
    name="set_plan",
    description="Generates and sets the initial step-by-step plan for solving the challenge.",
    parameters=SetPlan.model_json_schema()
)


all_tools = Tool(
    function_declarations=[
        # set_plan_fn,
        navigate_to_object_fn,
        submit_final_count_fn,
        submit_final_object_reference_fn,
        finish_instruction_following_fn,
    ]
)
