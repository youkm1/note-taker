UPDATE workflow_entity
SET
  active = 1,
  nodes = json_set(
    nodes,
    '$[1].parameters.url',
    '={{ ''http://stt:8000/stt?lang='' + ($json.body?.lang || ''ko'') }}',
    '$[1].parameters.bodyParameters.parameters',
    json('[{"parameterType":"formBinaryData","name":"audio","inputDataFieldName":"audio"}]'),
    '$[1].parameters.options.timeout',
    1800000,
    '$[2].parameters.contentType',
    'json',
    '$[2].parameters.specifyBody',
    'json',
    '$[2].parameters.jsonBody',
    '={{ JSON.stringify({
  text: $json.text,
  title: $node[''Webhook - Upload Audio''].json.body?.title || $node[''Webhook - Upload Audio''].json.body?.meeting_title || ''Meeting '' + new Date().toISOString().split(''T'')[0],
  metadata: {
    source: ''meeting'',
    language: $node[''Webhook - Upload Audio''].json.body?.lang || ''ko'',
    timestamp: new Date().toISOString(),
    speaker: $node[''Webhook - Upload Audio''].json.body?.speaker || ''unknown''
  }
}) }}',
    '$[2].parameters.options.timeout',
    600000,
    '$[4].parameters.jsCode',
    'const ragResponse = $node[''RAG - Ingest with Semantic Chunking''].json;
const transcript = $node[''STT - Whisper Transcription''].json;

return [{
  json: {
    success: true,
    transcript: transcript.text,
    rag_result: {
      chunks_stored: ragResponse.chunks_stored || 0,
      chunk_ids: ragResponse.ids || []
    },
    timestamp: new Date().toISOString()
  }
}];'
  ),
  connections = json('{
    "Webhook - Upload Audio": {
      "main": [[{"node": "STT - Whisper Transcription", "type": "main", "index": 0}]]
    },
    "STT - Whisper Transcription": {
      "main": [[{"node": "RAG - Ingest with Semantic Chunking", "type": "main", "index": 0}]]
    },
    "RAG - Ingest with Semantic Chunking": {
      "main": [[{"node": "Format Response", "type": "main", "index": 0}]]
    }
  }')
WHERE id = 'rrLrxdNryiBO6c8V';

UPDATE workflow_entity
SET active = 0
WHERE id = 'eXdtfnkOffreBYgg';

UPDATE workflow_entity
SET nodes = json_set(
  nodes,
  '$[1].parameters.contentType',
  'json',
  '$[1].parameters.specifyBody',
  'json',
  '$[1].parameters.jsonBody',
  '={{ JSON.stringify({
  question: $json.body?.question || $json.question
}) }}'
)
WHERE id = 'JROJ48TqtHiKAM25';
