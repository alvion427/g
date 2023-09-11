import openai
import g_config
import tiktoken
import yaml

openai.api_key = g_config.openai_key

ENCODING = tiktoken.get_encoding("cl100k_base")

class Message(object):
  def __init__(self, role, content, name=None):
    self.role = role
    self.name = name  # Used for function messages
    self.content = content
    self.tokens = None

  def getTokens(self):
    if not self.tokens:
      self.tokens = ENCODING.encode(self.content)
    return self.tokens

  def toDict(self):
    result = {"role": self.role, "content": self.content}
    if self.name:
      result["name"] = self.name
    return result

  @staticmethod
  def fromYaml(loader, node):
    mapping = loader.construct_mapping(node)
    return Message(mapping['role'], mapping['content'], mapping.get('name'))

  @staticmethod
  def toYaml(dumper, data):
    return dumper.represent_mapping('!Message', data.toDict())


yaml.constructor.SafeConstructor.add_constructor('!Message', Message.fromYaml)
yaml.add_constructor("!Message", Message.fromYaml)
yaml.add_representer(Message, Message.toYaml)


class LLMParameter(object):
  def __init__(self, name, type, description, required=False):
    self.name = name
    self.type = type
    self.description = description
    self.required = required

class LLMFunction(object):
  def __init__(self, name, description, parameters, implementation):
    self.name = name
    self.description = description
    self.parameters = parameters
    self.implementation = implementation

  def toDict(self):
    return {
      "name": self.name,
      "description": self.description,
      "parameters": {
          "type": "object",
          "properties": { p.name: {"type": p.type, "description": p.description} for p in self.parameters },
          "required": [p.name for p in self.parameters if p.required],
        },
    }

  def invoke(self, paramterString):
    return self.implementation(paramterString)

def clipMessages(messages, maxTokens):
  usedTokens = 0
  result = []
  for i in reversed(range(len(messages))):
    message = messages[i]
    tokens = message.getTokens()
    if usedTokens + len(tokens) <= maxTokens:
      result.insert(0, message)
      usedTokens += len(tokens)
    elif usedTokens == len(tokens):
      break
    else:
      # split the message
      tokensLeft = maxTokens - usedTokens
      splitTokens = tokens[len(tokens) - tokensLeft:]
      splitContent = ENCODING.decode(splitTokens)
      result.insert(0, Message(message.role, splitContent))
      usedTokens += len(splitTokens)
      break
  return result

def ask(
    query,
    context=[],
    systemPrompt="",
    model=None,
    functions=None,
    functionContext=None,
    maxTokens=-1,
    maxAttempts=3):

  # HACK!  We need a way to have the parameters in the config file be overwritten by command line parameters
  # before they are stored in defaults.  We should probably just add some cmd parsing to g_config.py.
  model = model or g_config.model

  if maxTokens > 0:
    numQueryTokens = len(ENCODING.encode(query))
    numSystemTokens = len(ENCODING.encode(systemPrompt or ""))
    maxContextTokens = maxTokens - numQueryTokens - numSystemTokens

    messages = clipMessages(context, maxContextTokens)
  else:
    messages = context.copy()

  if systemPrompt and len(context) > 0 and context[0].role != "system":
    messages = [Message("system", systemPrompt)] + messages

  messages.append(Message("user", query))

  def askRaw(messages, functions):
    lastError = None
    for attempt in range(maxAttempts):
      try:
        # OH SNAP!!!  It seems like openai processes queries DIFFERENTLY depending on whether or not you pass
        # functions.  For example, when passing all the code for the camel codebase and asking "how tall is
        # lebron james?" then the reply is correct (it just uses internet knowledge of course).  If I make the
        # exact same call, but this time passing in a useless function definition (in this case, the weather
        # one they use as an example) it responds with "I didn't find anything about lebron james' height in
        # the code".  It's WEIRD that this would have any effect, right?
        if functions:
          return openai.ChatCompletion.create(
            model=model,
            messages=[m.toDict() for m in messages],
            functions=[f.toDict() for f in functions],
            function_call="auto")
        else:
          return openai.ChatCompletion.create(
            model=model,
            messages=[m.toDict() for m in messages])
        break
      except openai.OpenAIError as e:
        lastError = e
        pass
      raise lastError

  response = askRaw(messages, functions)

  # if a function were returned as a result, invoke the function and then ask again
  # keep repeating until we get a message response
  while functions and response.choices[0].message.get("function_call"):
    name = response.choices[0].message.function_call["name"]
    paramsString = response.choices[0].message.function_call["arguments"]
    # search list of functions for one with .name == name
    f = next((f for f in functions if f.name == name), None)
    functionMessage = f.implementation(paramsString, functionContext)
    messages.append(Message("function", functionMessage, name=name))
    response = askRaw(messages, functions)

  responseMessage = response.choices[0].message.content
  messages.append(Message("assistant", responseMessage))

  return responseMessage, messages

def askStreaming(
    query,
    context=[],
    systemPrompt="",
    model=None,
    maxTokens=-1,
    maxAttempts=3):

  # HACK!  We need a way to have the parameters in the config file be overwritten by command line parameters
  # before they are stored in defaults.  We should probably just add some cmd parsing to g_config.py.
  model = model or g_config.model

  if maxTokens > 0:
    numQueryTokens = len(ENCODING.encode(query))
    numSystemTokens = len(ENCODING.encode(systemPrompt or ""))
    maxContextTokens = maxTokens - numQueryTokens - numSystemTokens

    messages = clipMessages(context, maxContextTokens)
  else:
    messages = context.copy()

  if systemPrompt and len(context) > 0 and context[0].role != "system":
    messages = [Message("system", systemPrompt)] + messages

  messages.append(Message("user", query))

  for response in openai.ChatCompletion.create(
    model=model,
    messages=[m.toDict() for m in messages],
    stream=True,
  ):
    if "content" in response.choices[0].delta:
      yield response.choices[0].delta.content
