CREATE TABLE sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES profiles(id) ON DELETE CASCADE NOT NULL,
  paper_title TEXT,
  paper_url TEXT,
  paper_path TEXT, -- URL to Supabase Storage bucket
  status TEXT DEFAULT 'created', -- 'created', 'processing', 'completed', 'error'
  llm_config JSONB, -- Stores provider, model, and API keys
  final_state JSONB, -- Stores the full LangGraph OverallState after completion
  errors JSONB DEFAULT '[]',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);