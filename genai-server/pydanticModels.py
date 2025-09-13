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
    label: str = Field(..., description="The unique string identifier for this object, e.g., '40_chair', '102_wall', '23_water cooler'.")
    # id: str = Field(..., description="The unique identifier for this object, e.g., 'chair_5'.")
    # type: str = Field(...,
    #                   description="The semantic label of the object, e.g., 'chair'.")
    # pose: List[float] = Field(
    #     ..., description="The [X, Y, Z] coordinate of the object's center in the map frame.")


class VLMRequest(BaseModel):
    """
    Represents the complete state-of-the-world payload sent from the ROS system
    to the VLM server for a single reasoning step.
    """
    challenge_question: str = Field(
        ..., description="The raw, natural language query or instruction from the user that the robot must solve.")
    tasks: List[Subtask] = Field(
        ..., description="The AI's current step-by-step plan. This shows what has been completed, what is in progress, and what is pending.")
    current_view: List[str] = Field(
        ..., description="The robot's immediate perception, containing a list of objects currently visible in its camera feed.")
    all_objects_discovered: List[str] = Field(
        ..., description="The robot's long-term memory, containing a summary of all unique objects found since the task began.")
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

class DescribePoint(BaseModel):
    """
    Analyzes the image to create a natural language description for a specific
    navigation point. Use this to label a visually selected target.
    """
    reasoning: str = Field(...,
                           description="A step-by-step explanation for why this location was chosen and why the description is accurate.")
    description: str = Field(...,
                           description="A verbose, natural language description of the object or location at the specified point, e.g., 'the potted plant in the corner'.")

# This is the corresponding FunctionDeclaration for the tool
describe_point_fn = FunctionDeclaration(
    name="describe_point",
    description="Creates a natural language description for a specific navigation point in the image.",
    parameters=DescribePoint.model_json_schema()
)

class NavigateToPoint(BaseModel):
    """
    Identifies a specific navigation goal in the provided image and returns its
    pixel coordinates. Use this to translate a visual target into a point.
    """
    reasoning: str = Field(...,
                           description="A step-by-step explanation for how the top points were chosen.")
    points: List[List[float]] = Field(...,
                             description="A list of 5 potential navigation points, sorted in decreasing order of probability, containing the [y, x] coordinates of the target point. The values MUST be normalized to a 0-1000 scale, where [0, 0] is the top-left corner of the image.")
# This is the corresponding FunctionDeclaration for the tool
navigate_to_point_fn = FunctionDeclaration(
    name="navigate_to_point",
    description="Selects a list of 5 visual points in the image to potentially navigate towards.",
    parameters=NavigateToPoint.model_json_schema()
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

class VerifyObjectExists(BaseModel):
    """
    Checks if a specific object ID exists in the robot's ground-truth semantic list.
    Use this as a confirmation step before submitting an answer.
    """
    reasoning: str = Field(...,
                           description="A step-by-step explanation for why you believe this specific object ID is the correct target.")
    target_id: str = Field(...,
                           description="The unique ID of the object you want to verify, e.g., '40_chair', '102_wall', '23_water cooler'.")

verify_object_exists_fn = FunctionDeclaration(
    name="verify_object_exists",
    description="",
    parameters=VerifyObjectExists.model_json_schema()
)

class GetVisualConfirmation(BaseModel):
    """
    Commands the robot to draw a bounding box for a hypothesized target object ID.
    This tool is used to get visual feedback to confirm if the object highlighted
    by the system correctly matches the requirements of the user's challenge question.
    """
    reasoning: str = Field(...,
                           description="Explain your hypothesis and what specific attribute or spatial relationship you are trying to confirm with this visual check.")
    target_id: str = Field(...,
                           description="The unique ID of the object to be highlighted with a bounding box for confirmation, e.g., '40_chair', '102_wall', '23_water cooler'.")

get_visual_confirmation_fn = FunctionDeclaration(
    name="get_visual_confirmation",
    description="Asks the system to draw a bounding box for a specific object ID to visually confirm it meets the user's requirements.",
    parameters=GetVisualConfirmation.model_json_schema()
)

class RequestFilteredView(BaseModel):
    """
    Call this if the current annotated image is too cluttered with irrelevant
    bounding boxes. This will request a new image with annotations only for the
    specified object labels.
    """
    reasoning: str = Field(...,
                           description="Explain why the current view is too cluttered and which objects are most important to focus on.")
    labels_to_keep: List[str] = Field(...,
                                     description="A list of the only object labels that should be annotated in the new image, e.g., '40_chair', '102_wall', '23_water cooler'.")

request_filtered_view_fn = FunctionDeclaration(
    name="request_filtered_view",
    description="Requests a new, less cluttered image with annotations for only a specific set of object labels.",
    parameters=RequestFilteredView.model_json_schema()
)

class SubmitFinalObjectReference(BaseModel):
    """
    Submits the final, verified object ID for an 'object_reference' type question.
    Only call this tool after 'verify_object_exists' has returned a success confirmation
    for the target object.
    """
    reasoning: str = Field(..., description="Explain why this specific object is the correct final answer, referencing visual evidence")
    answer: str = Field(
        ..., description="The unique string ID of the single target object, which MUST be from the list of objects returned by the 'verify_object_exists' tool.")


submit_final_object_reference_fn = FunctionDeclaration(
    name="submit_final_object_reference",
    description="Submits the final, verified object ID for an 'object_reference' type question.",
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

