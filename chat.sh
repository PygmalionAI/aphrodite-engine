#!/bin/bash

if [ $# -lt 2 ]; then
  echo "Usage: bash chat_completion.sh <model_name> <max_tokens> [<temperature>] [<stop_sequence>]"
  exit 1
fi

ENDPOINT="http://localhost:8000/v1/chat/completions"

MODEL_NAME="$1"
MAX_TOKENS="$2"
TEMPERATURE="${3:-1.0}"  # Default temperature is 1.0 if not provided
STOP_SEQUENCE="${4:-\\n###}"  # Default stop sequence is '\n###' if not provided

echo "Enter your conversation ('q' to quit):"
CONVERSATION=""

while true; do
  read -p "You: " USER_INPUT

  if [ "$USER_INPUT" == "q" ]; then
    echo "Exiting..."
    exit 0
  fi

  # Append user input to the conversation
  CONVERSATION="$CONVERSATION\n### Human: $USER_INPUT"

  DATA=$(cat <<EOF
{
  "model": "$MODEL_NAME",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "$CONVERSATION"}
  ],
  "max_tokens": $MAX_TOKENS,
  "temperature": $TEMPERATURE,
  "stop": ["$STOP_SEQUENCE"]
}
EOF
)

  RESPONSE=$(curl -s -X POST -H "Content-Type: application/json" \
                            -d "$DATA" \
                            $ENDPOINT)

  AI_REPLY=$(echo "$RESPONSE" | jq -r '.choices[0].message.content')

  # Remove any generated text after the stop sequence
  AI_REPLY=$(echo "$AI_REPLY" | sed -n "/$STOP_SEQUENCE/q;p")

  echo -e "\033[1;35mBot:\033[0m \033[1;32m$AI_REPLY\033[0m"
done
