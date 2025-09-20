import os
import socket

from twitchcompanion.logger import logger as MainLogger

logger = MainLogger.getChild(__name__)
logger.disabled = False


class TwitchClient:
    def __init__(self, channel: str = None):
        self.access_token = os.environ["TWITCH_ACCESS_TOKEN"]  # your personal token
        if not self.access_token.startswith("oauth:"):
            self.access_token = f"oauth:{self.access_token}"

        self.channel = channel

        self.bot_nick = "BOT"
        self.irc_host = "irc.chat.twitch.tv"
        self.irc_port = 6667
        self.sock = socket.socket()
        self.sock.connect((self.irc_host, self.irc_port))
        # login
        self.sock.send(f"PASS {self.access_token}\n".encode("utf-8"))
        self.sock.send(f"NICK {self.bot_nick}\n".encode("utf-8"))
        self.sock.send(f"JOIN #{self.channel}\n".encode("utf-8"))
        print("Twitch IRC client connected")

    def send_message(self, message: str):
        if self.channel is None:
            logger.error("Channel is not set. Cannot send message.")
            return
        self.sock.send(f"PRIVMSG #{self.channel} :{message}\n".encode("utf-8"))


def test():
    client = TwitchClient('etaneex')
    client.send_message("Hello, This is a massage")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    test()