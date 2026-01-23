CREATE TABLE messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID REFERENCES sessions(id) ON DELETE CASCADE NOT NULL,
  role TEXT NOT NULL, -- 'user', 'assistant', 'system'
  content TEXT NOT NULL,
  agent TEXT, -- Which agent sent the message (e.g., 'web_research')
  timestamp TIMESTAMPTZ DEFAULT NOW()
);