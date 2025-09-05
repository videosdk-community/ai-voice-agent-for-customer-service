
import asyncio, os
from videosdk.agents import Agent, AgentSession, CascadingPipeline, JobContext, RoomOptions, WorkerJob,ConversationFlow
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.openai import OpenAILLM
from videosdk.plugins.elevenlabs import ElevenLabsTTS
from typing import AsyncIterator

# Pre-downloading the Turn Detector model
pre_download_model()

class MyVoiceAgent(Agent):
    def __init__(self):
        super().__init__(instructions="""You are a knowledgeable and empathetic customer service representative powered by AI, dedicated to enhancing customer experiences. Your primary role is to assist customers by addressing their inquiries, resolving issues, and providing helpful information about our company's services and policies. Always communicate in a warm, patient, and respectful manner, ensuring customers feel heard and valued. Your interaction style should be proactive, guiding users towards effective solutions and clarifying any uncertainties they may have. You are well-versed in our account management processes, support procedures, and troubleshooting techniques. While you handle a wide range of queries, direct sensitive information handling, such as financial details, to secure channels or human representatives as appropriate. Ensure each interaction is efficient, informative, and leaves a positive impression of our brand.""")
    async def on_enter(self): await self.session.say("Hello! How can I help you today regarding ai voice agent for customer service?")
    async def on_exit(self): await self.session.say("Goodbye!")

async def start_session(context: JobContext):
    # Create agent and conversation flow
    agent = MyVoiceAgent()
    conversation_flow = ConversationFlow(agent)

    # Create pipeline
    pipeline = CascadingPipeline(
        stt=DeepgramSTT(model="nova-2", language="en"),
        llm=OpenAILLM(model="gpt-4o"),
        tts=ElevenLabsTTS(model="eleven_flash_v2_5"),
        vad=SileroVAD(threshold=0.35),
        turn_detector=TurnDetector(threshold=0.8)
    )

    session = AgentSession(
        agent=agent,
        pipeline=pipeline,
        conversation_flow=conversation_flow
    )

    try:
        await context.connect()
        await session.start()
        # Keep the session running until manually terminated
        await asyncio.Event().wait()
    finally:
        # Clean up resources when done
        await session.close()
        await context.shutdown()

def make_context() -> JobContext:
    room_options = RoomOptions(
     #  room_id="YOUR_MEETING_ID",  # Set to join a pre-created room; omit to auto-create
        name="VideoSDK Cascaded Agent for ai voice agent for customer service",
        playground=True
    )

    return JobContext(room_options=room_options)

if __name__ == "__main__":
    job = WorkerJob(entrypoint=start_session, jobctx=make_context)
    job.start()
