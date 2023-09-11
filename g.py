import argparse
import sys
import os
import yaml
from datetime import datetime

import g_config
import llm

def getDataDir():
  if g_config.data_dir:
    return g_config.data_dir
  else:
    return os.path.expanduser("~/.gee")

def listStreams():
  dataDir = getDataDir()
  if os.path.isdir(dataDir):
    streamsDir = os.path.join(dataDir, 'streams')
    for s in os.listdir(streamsDir):
      if os.path.isfile(os.path.join(streamsDir, s)):
        yield os.path.splitext(s)[0]


def streamExists(stream):
  dataDir = getDataDir()
  streamFile = os.path.join(dataDir, 'streams', f'{stream}.txt')
  streamFile = os.path.normpath(streamFile)
  return os.path.isfile(streamFile)

def readStream(stream):
  dataDir = getDataDir()
  streamFile = os.path.join(dataDir, 'streams', f'{stream}.txt')

  streamFile = os.path.normpath(streamFile)

  if not os.path.isfile(streamFile):
    return []

  with open(streamFile, 'r') as f:
    text = f.read()
    content = yaml.safe_load(text)

    def parseMessage(item):
      role = next(iter(item))
      content = item[role]
      return llm.Message(role, content)

    return [parseMessage(item) for item in content]


def saveMessages(path, messages):
  contents = [{item.role: item.content} for item in messages]
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, 'w') as f:
    f.write(yaml.dump(contents))

def saveStream(stream, messages):
  dataDir = getDataDir()
  streamFile = os.path.join(dataDir, 'streams', f'{stream}.txt')
  saveMessages(streamFile, messages)

def parseArgs():
  parser = argparse.ArgumentParser(description='Generate text with GPT-3/4')

  parser.add_argument('--list', action='store_true', dest='streams', default=False, help='List all streams')
  parser.add_argument('-s', '--stream', action='store', dest='stream', default=None, help='Stream to use for communication')
  parser.add_argument('-sys', '--system', action='store', dest='system', default=None, help='System message.')
  parser.add_argument('-m', '--model', action='store', dest='model', default=g_config.model, type=str, help='Use specific model')
  parser.add_argument('-l', '--log', action='store_true', dest='log', default=True, help='Whether or not to log requests')
  parser.add_argument('-i', '--interactive', action='store_true', default=False, help='Continue prompting after first reply.  Defaults to true unless you pass an initial prompt')

  # parse all remaining positional arguments as a list
  parser.add_argument('args', nargs=argparse.REMAINDER)

  args = parser.parse_args()

  return args

def chat(args, messages, docType, initialPrompt="", onlyOnce=False):
  functions = docType and docType.functions or None

  while True:
    if initialPrompt:
      prompt = initialPrompt
      initialPrompt = None
    else:
      try:
        prompt = input("\nEnter prompt: ")
      except KeyboardInterrupt:
        return

    if prompt in ["quit", "exit"]:
      print("Goodbye!")
      break

    fullResponse = ""
    try:
      for partialResponse in llm.askStreaming(prompt, messages):
        print(partialResponse, end="")
        fullResponse += partialResponse
    except KeyboardInterrupt:
      onlyOnce = True

    messages.append(llm.Message("assistant", fullResponse))

    if args.stream:
      saveStream(args.stream, messages)

    if args.log:
      # Generate log file name from timestamp
      logFile = os.path.join(getDataDir(), 'logs', f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.txt')
      os.makedirs(os.path.dirname(logFile), exist_ok=True)
      contents = "\n".join([f"*** ROLE: {m.role} ***\n\n{m.content}" for m in messages])
      with open(logFile, 'w') as f:
        f.write(contents)

    if onlyOnce:
      break

def main():
  args = parseArgs()

  if args.model:
    g_config.model = args.model

  if args.streams:
    print("Streams:")
    for s in listStreams():
      print(s)
    return

  initialPrompt = ' '.join(args.args)
  messages = []

  onlyOnce = not args.interactive and len(initialPrompt) > 0

  docType = None
  if args.stream:
    # streams are stored as a list of role/content pairs
    print(f"Reading stream from {args.stream}.txt")
    messages = readStream(args.stream)

  chat(args, messages, docType, initialPrompt=initialPrompt, onlyOnce=onlyOnce)

if __name__ == '__main__':
  main()
