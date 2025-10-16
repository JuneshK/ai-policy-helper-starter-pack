'use client';
import React, { useState } from 'react';
import { MessageSquareText, MessageCircle, Send } from 'lucide-react';
import { apiAsk } from '../lib/api';

type Message = {
  role: 'user' | 'assistant';
  content: string;
  citations?: { title: string; section?: string }[];
  chunks?: { title: string; section?: string; text: string }[];
};

// ---------------- Subcomponents ---------------- //
const EmptyState = () => (
  <div className="flex flex-col items-center justify-center h-full text-center text-gray-400">
    <MessageSquareText className="w-12 h-12 mb-3 text-indigo-300" />
    <p className="text-lg font-medium mb-1">Start the policy conversation</p>
    <p className="text-sm">Ask about returns, shipping, or product details.</p>
  </div>
);

const Citations = ({ citations }: { citations: Message['citations'] }) => (
  <div className="flex flex-wrap gap-2 mt-1">
    {citations?.map((c, idx) => (
      <span
        key={idx}
        className="bg-indigo-100 text-indigo-800 text-xs px-2 py-1 rounded-full cursor-help"
        title={c.section || ''}
      >
        {c.title}
      </span>
    ))}
  </div>
);

const SupportingChunks = ({ chunks }: { chunks: Message['chunks'] }) => (
  <details className="mt-2 bg-gray-50 p-3 rounded-lg border border-gray-200">
    <summary className="cursor-pointer text-indigo-600 font-medium">
      View supporting chunks
    </summary>
    <div className="mt-2 space-y-2">
      {chunks?.map((c, idx) => (
        <div key={idx} className="bg-gray-100 p-3 rounded-lg shadow-sm">
          <div className="font-semibold text-gray-800">
            {c.title}
            {c.section ? ' — ' + c.section : ''}
          </div>
          <div className="whitespace-pre-wrap text-gray-700 text-sm">{c.text}</div>
        </div>
      ))}
    </div>
  </details>
);

const MessageItem = ({ message }: { message: Message }) => {
  const isUser = message.role === 'user';
  return (
    <div className="flex flex-col space-y-1">
      <div className="text-xs text-gray-500">{isUser ? 'You' : 'Assistant'}</div>
      <div
        className={`p-3 rounded-xl max-w-[80%] shadow-sm self-${
          isUser
            ? 'end bg-indigo-600 text-white rounded-br-none'
            : 'start bg-gray-100 text-gray-800 rounded-tl-none'
        }`}
      >
        {message.content}
      </div>
      {message.citations && <Citations citations={message.citations} />}
      {message.chunks && <SupportingChunks chunks={message.chunks} />}
    </div>
  );
};

// ---------------- Main Chat Component ---------------- //
export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [q, setQ] = useState('');
  const [loading, setLoading] = useState(false);

  const send = async () => {
    if (!q.trim()) return;

    const userMessage: Message = { role: 'user', content: q };
    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);

    try {
      const res = await apiAsk(q);
      const aiMessage: Message = {
        role: 'assistant',
        content: res.answer,
        citations: res.citations,
        chunks: res.chunks,
      };
      setMessages((prev) => [...prev, aiMessage]);
    } catch (e: any) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Error: ' + e.message },
      ]);
    } finally {
      setLoading(false);
      setQ('');
    }
  };

  return (
    <div className="bg-white p-6 rounded-xl shadow-xl flex flex-col h-[500px] transition-all">
      {/* Header */}
      <div className="flex items-center mb-4 space-x-3">
        <MessageCircle className="w-9 h-10 mb-3 text-indigo-600" />
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Chat</h2>
      </div>

      {/* Messages Container */}
      <div className="flex-grow overflow-y-auto p-6 space-y-4">
        {messages.length === 0 ? (
          <EmptyState />
        ) : (
          messages.map((m, i) => <MessageItem key={i} message={m} />)
        )}
      </div>

      {/* Input Area */}
      <div className="flex gap-3 mt-auto bg-gray-100 rounded-lg p-2">
        <input
          className="flex-1 p-3 rounded-lg border border-gray-600 text-black focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
          placeholder="Ask about policy or products..."
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && send()}
        />
        <button
          onClick={send}
          disabled={loading}
          className={`px-5 py-3 rounded-lg font-semibold transition duration-200 shadow-md ${
            loading ? 'bg-indigo-300 cursor-not-allowed' : 'bg-indigo-600 text-white'
          }`}
        >
          {loading ? 'Thinking...' : <Send className="w-5 h-5" />}
        </button>
      </div>
    </div>
  );
}


