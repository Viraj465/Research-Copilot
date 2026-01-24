-- WARNING: This schema is for context only and is not meant to be run.
-- Table order and constraints may not be valid for execution.

CREATE TABLE public.deep_dive_messages (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  session_id uuid NOT NULL,
  field text NOT NULL,
  role text NOT NULL,
  content text NOT NULL,
  timestamp timestamp with time zone DEFAULT now(),
  CONSTRAINT deep_dive_messages_pkey PRIMARY KEY (id),
  CONSTRAINT deep_dive_messages_session_id_fkey FOREIGN KEY (session_id) REFERENCES public.sessions(id)
);
CREATE TABLE public.messages (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  session_id uuid NOT NULL,
  role text NOT NULL,
  content text NOT NULL,
  agent text,
  timestamp timestamp with time zone DEFAULT now(),
  CONSTRAINT messages_pkey PRIMARY KEY (id),
  CONSTRAINT messages_session_id_fkey FOREIGN KEY (session_id) REFERENCES public.sessions(id)
);
CREATE TABLE public.profiles (
  id uuid NOT NULL,
  email text UNIQUE,
  full_name text,
  avatar_url text,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  CONSTRAINT profiles_pkey PRIMARY KEY (id),
  CONSTRAINT profiles_id_fkey FOREIGN KEY (id) REFERENCES auth.users(id)
);
CREATE TABLE public.sessions (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  paper_title text,
  paper_url text,
  paper_path text,
  status text DEFAULT 'created'::text,
  llm_config jsonb,
  final_state jsonb,
  errors jsonb DEFAULT '[]'::jsonb,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  CONSTRAINT sessions_pkey PRIMARY KEY (id),
  CONSTRAINT sessions_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.profiles(id)
);