"""
Chat with BayLing with command line interface.

Usage:
python chat.py --model-path ${PATH_TO_BAYLING} --style rich --load-8bit
"""
import argparse
import os
import re
import sys

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live

try:
    from fastchat.serve.inference import chat_loop
except:
    chat_loop = None

from bayling.model_adapter import add_model_args
from bayling.inference import ChatIO
from bayling.inference import chat_loop as bayling_chat_loop
import warnings
import pdb
warnings.filterwarnings("ignore")


class SimpleChatIO(ChatIO):
    def prompt_for_input(self, role) -> str:
        return input(f"USER:\n")

    def prompt_for_output(self, role: str):
        print(f"FinGPT:\n", end="", flush=True)

    def stream_output(self, output_stream):
        pre = 0
        for outputs in output_stream:
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                print(" ".join(output_text[pre:now]), end=" ", flush=True)
                pre = now
        print(" ".join(output_text[pre:]), flush=True)
        return " ".join(output_text)


class RichChatIO(ChatIO):
    def __init__(self):
        self._prompt_session = PromptSession(history=InMemoryHistory())
        self._completer = WordCompleter(
            words=["!exit", "!reset"], pattern=re.compile("$")
        )
        self._console = Console()

        style = "bold white on blue"
        self._console.print(
            "Welcome to interact with FinGPT!", style=style, justify="center"
        )
        self._console.print(
            "- Enter your question and press 'Enter' to interact with FinGPT.",
            style="white on blue",
            justify="left",
        )
        self._console.print(
            "- Enter 'q' to clear the interaction history.",
            style="white on blue",
            justify="left",
        )

    def prompt_for_input(self, role) -> str:
        self._console.print(f"[bold]USER:\n", end="")
        # TODO(suquark): multiline input has some issues. fix it later.
        prompt_input = self._prompt_session.prompt(
            completer=self._completer,
            multiline=False,
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=None,
        )
        self._console.print()
        return prompt_input

    def prompt_for_output(self, role: str):
        self._console.print(f"[bold]FinGPT:\n", end="")

    def stream_output(self, output_stream):
        """Stream output from a role."""
        # TODO(suquark): the console flickers when there is a code block
        #  above it. We need to cut off "live" when a code block is done.

        # Create a Live context for updating the console output
        with Live(console=self._console, refresh_per_second=4) as live:
            # Read lines from the stream
            for outputs in output_stream:
                if not outputs:
                    continue
                text = outputs["text"]
                # Render the accumulated text as Markdown
                # NOTE: this is a workaround for the rendering "unstandard markdown"
                #  in rich. The chatbots output treat "\n" as a new line for
                #  better compatibility with real-world text. However, rendering
                #  in markdown would break the format. It is because standard markdown
                #  treat a single "\n" in normal text as a space.
                #  Our workaround is adding two spaces at the end of each line.
                #  This is not a perfect solution, as it would
                #  introduce trailing spaces (only) in code block, but it works well
                #  especially for console output, because in general the console does not
                #  care about trailing spaces.
                lines = []
                for line in text.splitlines():
                    lines.append(line)
                    if line.startswith("```"):
                        # Code block marker - do not add trailing spaces, as it would
                        #  break the syntax highlighting
                        lines.append("\n")
                    else:
                        lines.append("  \n")
                markdown = Markdown("".join(lines))
                # Update the Live console output
                live.update(markdown)
        self._console.print()
        return text


def main(args):
    if args.gpus:
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    if args.style == "simple":
        chatio = SimpleChatIO()
    elif args.style == "rich":
        chatio = RichChatIO()
    else:
        raise ValueError(f"Invalid style for console: {args.style}")
    try:
        is_bayling = (
            "bayling" in args.model_path.lower() or "imt" in args.model_path.lower()
        )
        #fix(chat.py)here is the bug, set is_bayling -> True 
        is_bayling = True

        if is_bayling:
            chat_loop_func = bayling_chat_loop
        else:
            assert chat_loop is not None
            chat_loop_func = chat_loop
        

        chat_loop_func(
            args.model_path,
            args.device,
            args.num_gpus,
            args.max_gpu_memory,
            args.load_8bit,
            args.cpu_offloading,
            args.conv_template,
            args.temperature,
            args.repetition_penalty,
            args.max_new_tokens,
            chatio,
            args.debug,
        )
    except KeyboardInterrupt:
        print("exit...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--style",
        type=str,
        default="rich",
        choices=["simple", "rich"],
        help="Display style.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print useful debug information (e.g., prompts)",
    )
    args = parser.parse_args()
    main(args)
