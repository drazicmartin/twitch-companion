import time
from twitchcompanion.main import TwitchWatcher
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from dataclasses import dataclass
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, SystemMessage

SYSTEM_PROMPT = """You are a stream companion. 
Your role is to interact with the streamer and the viewers in a fun and engaging way. 
You must write the answer in French. 
Your name is NIOX; the streamer can ask you specific things by calling you NIOX, and you must answer accordingly. 
The title of the stream is {title}. 
The streamer name is {channel}. He is currently playing {category}. 
You must interact with the streamer based on what he is saying. Keep your responses short and engaging. 
You can make jokes and be funny or even silly. Do not mention you are an AI model or chatbot. 
Caution: some of the things in the transcription may be inaccurate or come from the background music."""

@dataclass
class Context:
    """Custom runtime context schema."""
    channel: str
    game_name: str

@tool
def wait():
    """Wait for a short period before responding again. allowing more transcriptions to accumulate."""
    return


class TwitchAgent(TwitchWatcher):
    def __init__(self, channel: str, **kwargs):
        super().__init__(channel, **kwargs)

        self.llm = init_chat_model(
            "anthropic:claude-sonnet-4-5-20250929",
            temperature=0.5,
            max_tokens=50,
        )

        self.count_line_read = 0
        self.agent = None

    def init_agent(self):
        checkpointer = InMemorySaver()
        system_prompt = SYSTEM_PROMPT.format(
            title=self.title,
            channel=self.channel,
            category=self.category
        )
        self.agent = create_agent(
            model=self.llm,
            system_prompt=system_prompt,
            tools=[wait],
            context_schema=Context,
            # response_format=ResponseFormat,
            checkpointer=checkpointer
        )
        self.config = {"configurable": {"thread_id": "1"}}
    
    def start_workers(self, whisper_model_size="medium"):
        super().start_workers(whisper_model_size=whisper_model_size)
        self.init_agent()

    def _response_main(self): 
        """Invoke the agent with the current context."""
        if not self.should_respond():
            return  # too soon to respond again

        context = Context(
            channel=self.channel,
            game_name=self.category
        )

        transcription = self.transcriber.get_latest_transcription(n=0) # get all lines
        transcription = transcription[self.count_line_read:]           # only new lines
        response = self.agent.invoke(
            {"messages": [HumanMessage(line) for line in transcription]},
            context=context,
            config=self.config,
        )
        self.count_line_read += len(transcription)
        content = response['messages'][-1].content
        if isinstance(content, list):
            breakpoint()
            message = content[0].get('text', '').strip()
        else:
            message = str(content).strip()
        
        self.handle_send(message)

def test():
    agent = TwitchAgent(channel="etaneex")
    agent.init_agent()
    response = agent._response_main()
    print(response)

if __name__ == "__main__":
    test()