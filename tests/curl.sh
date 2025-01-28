
bearer=$1
query=$2
sess=$3
req=$4
port=8001
user=cb

curl -X POST localhost:$port/api/thirdbrain-mcp-openai-agent \
-H "Authorization: Bearer '$bearer'" \
-H "Content-Type: application/json" \
-d '{
  "query": "'$query'",
  "user_id": "'$user'",
  "request_id": "R123_'$req'",
  "session_id": "'$sess'",
  "files": []
}'