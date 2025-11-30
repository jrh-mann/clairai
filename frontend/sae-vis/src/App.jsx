import React, { useState, useEffect, useMemo, useRef } from 'react';
import { Send, Settings, Activity, Cpu } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

// --- UTILS ---
function cn(...inputs) {
  return twMerge(clsx(inputs));
}

// --- CONFIG ---
const PROXY_URL = "http://localhost:6969/v1/chat/completions";
// Ensure this matches your vLLM model ID exactly
const MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"; 

// --- COMPONENTS ---

const TokenRenderer = ({ 
  text, 
  score, 
  maxScore, 
  index, 
  hoveredIndex, 
  setHoveredIndex,
  selectedProbe
}) => {
  // Calculate intensity (0 to 1). SAE features are ReLU (positive).
  // We use a non-linear scaling (sqrt) to make lower values more visible.
  const intensity = maxScore > 0 ? Math.sqrt(Math.max(0, score) / maxScore) : 0;
  
  // Color: Emerald Green (Tailwind-ish). 
  // We apply opacity via RGBA.
  // r=16, g=185, b=129 is roughly emerald-500
  const backgroundColor = `rgba(16, 185, 129, ${intensity * 0.8})`; 
  
  const [showTooltip, setShowTooltip] = useState(false);
  
  return (
    <span
      className="relative inline-block"
      onMouseEnter={() => {
        setHoveredIndex(index);
        setShowTooltip(true);
      }}
      onMouseLeave={() => {
        setHoveredIndex(null);
        setShowTooltip(false);
      }}
    >
      <span
        className={cn(
          "inline-block px-0.5 rounded transition-colors duration-75 cursor-crosshair border-b-2",
          hoveredIndex === index 
            ? "border-black bg-yellow-200" // Hover style overrides heat map
            : "border-transparent"
        )}
        style={{ 
          backgroundColor: hoveredIndex === index ? undefined : backgroundColor,
          minWidth: text === " " ? "0.25em" : undefined
        }}
      >
        {text}
      </span>
      
      {/* Tooltip */}
      {showTooltip && (
        <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 z-50 pointer-events-none">
          <div className="bg-black text-white text-xs px-3 py-2 rounded shadow-lg whitespace-nowrap">
            <div>Token {index}</div>
            <div>Probe {selectedProbe}: {(score ?? 0).toFixed(4)}</div>
          </div>
          {/* Arrow */}
          <div className="absolute top-full left-1/2 transform -translate-x-1/2 -mt-1">
            <div className="border-4 border-transparent border-t-black"></div>
          </div>
        </div>
      )}
    </span>
  );
};

const ProbeGraph = ({ data, selectedProbe, hoveredIndex, setHoveredIndex, numProbes }) => {
  if (!data || data.length === 0) return null;

  // Prepare lines for the chart
  // We highlight the selected probe, ghost the others
  return (
    <div className="h-48 w-full bg-white border-t border-gray-200 p-4 shadow-up transition-all">
      <div className="flex justify-between mb-2">
        <h3 className="text-xs font-bold uppercase text-gray-500 tracking-wider flex items-center gap-2">
          <Activity size={14} /> Probe Activations
        </h3>
        <span className="text-xs text-gray-400">
          Token {hoveredIndex !== null ? hoveredIndex : '-'}
        </span>
      </div>
      
      <ResponsiveContainer width="100%" height="100%">
        <LineChart 
          data={data}
          onMouseMove={(e) => {
            if (e.activeTooltipIndex !== undefined) {
              setHoveredIndex(e.activeTooltipIndex);
            }
          }}
          onMouseLeave={() => setHoveredIndex(null)}
        >
          <XAxis 
            dataKey="index" 
            hide 
          />
          <YAxis 
            domain={[0, 'auto']}
            label={{ value: 'Activation', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle', fontSize: '12px', fill: '#666' } }}
            style={{ fontSize: '11px' }}
            width={60}
          />
          <Tooltip 
            trigger="hover"
            content={({ active, payload, label }) => {
              if (active && payload && payload.length) {
                // Find selected probe value - the dataKey is "probe{selectedProbe}"
                const probeKey = `probe${selectedProbe}`;
                const selectedPayload = payload.find(p => p.dataKey === probeKey || p.name === probeKey);
                const val = selectedPayload?.value ?? selectedPayload?.payload?.[probeKey];
                
                return (
                  <div className="bg-black text-white text-xs px-3 py-2 rounded shadow-lg">
                    <div className="font-semibold mb-1">Token {label}</div>
                    <div>Probe {selectedProbe}: {val !== undefined && val !== null ? Number(val).toFixed(4) : 'N/A'}</div>
                  </div>
                );
              }
              return null;
            }}
          />
          
          {/* Render faint lines for context (first 5 probes only to save perfo if N is huge) */}
          {Array.from({ length: Math.min(numProbes, 5) }).map((_, i) => 
            i !== selectedProbe ? (
              <Line 
                key={i} 
                type="monotone" 
                dataKey={`probe${i}`} 
                stroke="#e5e7eb" 
                strokeWidth={1} 
                dot={false} 
                isAnimationActive={false}
              />
            ) : null
          )}

          {/* Render Selected Probe */}
          <Line 
            type="monotone" 
            dataKey={`probe${selectedProbe}`}
            name={`probe${selectedProbe}`}
            stroke="#10b981" 
            strokeWidth={2} 
            dot={false}
            activeDot={{ r: 4, fill: 'black' }}
            isAnimationActive={false}
          />
          
          {/* Sync Line from Text Hover */}
          {hoveredIndex !== null && (
            <ReferenceLine x={hoveredIndex} stroke="black" strokeDasharray="3 3" />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default function App() {
  const [input, setInput] = useState("List 3 colors.");
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [messageIdCounter, setMessageIdCounter] = useState(0);
  
  // SAE Interpretation State
  const [selectedProbe, setSelectedProbe] = useState(0);
  const [hoveredTokenIndex, setHoveredTokenIndex] = useState(null);
  const [numProbes, setNumProbes] = useState(0);

  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMsg = { id: messageIdCounter, role: 'user', content: input };
    const assistantMsgId = messageIdCounter + 1;
    setMessageIdCounter(assistantMsgId + 1);
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);

    // Create placeholder assistant message that we'll update as tokens stream in
    const assistantMsg = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      tokens: [],
      scores: []
    };
    setMessages(prev => [...prev, assistantMsg]);

    try {
      const response = await fetch(PROXY_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: MODEL_NAME,
          messages: [...messages, userMsg].map(m => ({ role: m.role, content: m.content })),
          max_tokens: 100 // Short for demo
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6);
            if (dataStr === '[DONE]') continue;
            
            try {
              const data = JSON.parse(dataStr);
              
              if (data.type === 'token') {
                // Stream token in real-time - create new object to trigger React update
                setMessages(prev => prev.map(msg => 
                  msg.id === assistantMsgId 
                    ? {
                        ...msg,
                        content: msg.content + data.token,
                        tokens: [...msg.tokens, data.token]
                      }
                    : msg
                ));
              } else if (data.type === 'probe_update' || data.type === 'probe_final') {
                // Incremental or final probe data update
                if (data.probe_scores) {
                  // Update global probe count if first run
                  if (data.probe_scores.length > 0) {
                    setNumProbes(data.probe_scores.length);
                  }
                  
                  // Update the message in state with new probe data
                  setMessages(prev => prev.map(msg => 
                    msg.id === assistantMsgId 
                      ? {
                          ...msg,
                          scores: data.probe_scores
                        }
                      : msg
                  ));
                }
              } else if (data.type === 'done') {
                // Final message - ensure we have latest probe data
                if (data.probe_scores && data.probe_scores.length > 0) {
                  setNumProbes(data.probe_scores.length);
                }
                
                // Final update
                setMessages(prev => prev.map(msg => 
                  msg.id === assistantMsgId 
                    ? {
                        ...msg,
                        content: data.content || msg.content,
                        tokens: data.tokens || msg.tokens,
                        scores: data.probe_scores || msg.scores || []
                      }
                    : msg
                ));
              }
            } catch (e) {
              // Skip malformed JSON
              console.warn('Failed to parse SSE data:', dataStr);
            }
          }
        }
      }
    } catch (err) {
      console.error(err);
      setMessages(prev => prev.map(msg => 
        msg.id === assistantMsgId
          ? { ...msg, content: msg.content || "Error connecting to Proxy." }
          : msg
      ));
    } finally {
      setIsLoading(false);
    }
  };

  // Prepare Graph Data for the LATEST assistant message only
  const lastMessage = messages.filter(m => m.role === 'assistant').slice(-1)[0];
  
  const graphData = useMemo(() => {
    if (!lastMessage || !lastMessage.scores || lastMessage.scores.length === 0) return [];
    
    // Transform from [n_probes, n_tokens] -> Array of { index, probe0, probe1... }
    const numTokens = lastMessage.tokens.length;
    const chartData = [];

    for (let t = 0; t < numTokens; t++) {
      const point = { index: t };
      // Flatten scores for recharts
      lastMessage.scores.forEach((probeRow, probeIdx) => {
        point[`probe${probeIdx}`] = probeRow[t];
      });
      chartData.push(point);
    }
    return chartData;
  }, [lastMessage]);

  // Max score for the currently selected probe (for normalization)
  const currentMaxScore = useMemo(() => {
    if (!lastMessage || !lastMessage.scores) return 1;
    const probeRow = lastMessage.scores[selectedProbe];
    if (!probeRow) return 1;
    return Math.max(...probeRow);
  }, [lastMessage, selectedProbe]);

  return (
    <div className="flex flex-col h-screen bg-gray-50 font-sans text-gray-900">
      
      {/* --- HEADER & CONTROLS --- */}
      <header className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between sticky top-0 z-10 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="bg-emerald-100 p-2 rounded-lg">
            <Cpu className="text-emerald-600" size={20} />
          </div>
          <div>
            <h1 className="font-bold text-lg leading-tight">SAE Microscope</h1>
            <p className="text-xs text-gray-500">vLLM Probe Visualization</p>
          </div>
        </div>

        {/* Probe Selector */}
        <div className="flex items-center gap-4 bg-gray-100 p-1.5 rounded-lg border border-gray-200">
          <span className="text-xs font-semibold uppercase text-gray-500 pl-2">Feature / Probe</span>
          <div className="flex items-center bg-white rounded shadow-sm border border-gray-200">
             <button 
               onClick={() => setSelectedProbe(Math.max(0, selectedProbe - 1))}
               className="px-3 py-1 hover:bg-gray-50 border-r border-gray-100 text-gray-600"
               disabled={selectedProbe === 0}
             >-</button>
             <input 
                type="number" 
                value={selectedProbe}
                onChange={(e) => setSelectedProbe(Math.max(0, parseInt(e.target.value) || 0))}
                className="w-16 text-center text-sm font-mono py-1 outline-none"
             />
             <button 
               onClick={() => setSelectedProbe(selectedProbe + 1)}
               className="px-3 py-1 hover:bg-gray-50 border-l border-gray-100 text-gray-600"
             >+</button>
          </div>
          {numProbes > 0 && <span className="text-xs text-gray-400 pr-2">of {numProbes}</span>}
        </div>
      </header>

      {/* --- CHAT AREA --- */}
      <main className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.length === 0 && (
          <div className="h-full flex items-center justify-center text-gray-400 text-sm">
            Ready to probe. Send a message to inspect features.
          </div>
        )}

        {messages.map((msg, i) => {
          // Check if this is the last assistant message (the one we're visualizing)
          const isLastAssistantMessage = msg.role === 'assistant' && 
            i === messages.length - 1 && 
            msg.tokens && 
            msg.tokens.length > 0;
          
          // Only apply hover highlighting to the last assistant message
          const effectiveHoveredIndex = isLastAssistantMessage ? hoveredTokenIndex : null;
          
          return (
            <div key={msg.id || i} className={cn("flex", msg.role === 'user' ? "justify-end" : "justify-start")}>
              <div className={cn(
                "max-w-3xl rounded-2xl p-5 shadow-sm text-base leading-relaxed",
                msg.role === 'user' 
                  ? "bg-blue-600 text-white rounded-br-none" 
                  : "bg-white border border-gray-200 rounded-bl-none"
              )}>
                {msg.role === 'user' ? (
                  msg.content
                ) : (
                  <div className="font-mono text-sm">
                    {/* If we have probe data, render tokens. Else plain text. */}
                    {msg.tokens && msg.tokens.length > 0 ? (
                      <div className="flex flex-wrap items-baseline content-start gap-y-1">
                        {msg.tokens.map((token, tIdx) => (
                          <TokenRenderer
                            key={tIdx}
                            text={token}
                            index={tIdx}
                            score={msg.scores[selectedProbe]?.[tIdx] || 0}
                            maxScore={currentMaxScore}
                            hoveredIndex={effectiveHoveredIndex}
                            setHoveredIndex={isLastAssistantMessage ? setHoveredTokenIndex : () => {}}
                            selectedProbe={selectedProbe}
                          />
                        ))}
                      </div>
                    ) : (
                      <span>{msg.content}</span>
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}
        <div ref={messagesEndRef} />
      </main>

      {/* --- FOOTER: GRAPH & INPUT --- */}
      <div className="bg-white border-t border-gray-200 z-10">
        
        {/* GRAPH PANEL */}
        {graphData.length > 0 && (
          <ProbeGraph 
            data={graphData} 
            selectedProbe={selectedProbe} 
            hoveredIndex={hoveredTokenIndex}
            setHoveredIndex={setHoveredTokenIndex}
            numProbes={numProbes}
          />
        )}

        {/* INPUT AREA */}
        <form onSubmit={handleSubmit} className="p-4 bg-gray-50 border-t border-gray-200">
          <div className="relative max-w-4xl mx-auto flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your message..."
              disabled={isLoading}
              className="flex-1 p-3 rounded-lg border border-gray-300 shadow-sm focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none transition-all disabled:opacity-50"
            />
            <button 
              type="submit" 
              disabled={isLoading}
              className="bg-emerald-600 hover:bg-emerald-700 text-white px-6 rounded-lg font-medium transition-colors disabled:opacity-50 flex items-center gap-2 shadow-sm"
            >
              {isLoading ? <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" /> : <Send size={18} />}
              Send
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
