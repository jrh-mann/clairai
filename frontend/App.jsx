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
  setHoveredIndex
}) => {
  // Calculate intensity (0 to 1). SAE features are ReLU (positive).
  // We use a non-linear scaling (sqrt) to make lower values more visible.
  const intensity = maxScore > 0 ? Math.sqrt(Math.max(0, score) / maxScore) : 0;
  
  // Color: Emerald Green (Tailwind-ish). 
  // We apply opacity via RGBA.
  // r=16, g=185, b=129 is roughly emerald-500
  const backgroundColor = `rgba(16, 185, 129, ${intensity * 0.8})`; 
  
  return (
    <span
      onMouseEnter={() => setHoveredIndex(index)}
      onMouseLeave={() => setHoveredIndex(null)}
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
          <XAxis dataKey="index" hide />
          <YAxis hide domain={[0, 'auto']} />
          <Tooltip 
            trigger="hover"
            content={({ active, payload, label }) => {
              if (active && payload && payload.length) {
                // Find selected probe value
                const val = payload.find(p => p.name === `Probe ${selectedProbe}`)?.value;
                return (
                  <div className="bg-black text-white text-xs p-2 rounded shadow-lg">
                    <p>Token: {label}</p>
                    <p>Probe {selectedProbe}: {val?.toFixed(4)}</p>
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

    const userMsg = { role: 'user', content: input };
    const assistantMsg = {
      role: 'assistant',
      content: '',
      tokens: [],
      scores: [] // shape: [n_probes][n_tokens]
    };
    
    // Add both messages in a single state update
    setMessages(prev => [...prev, userMsg, assistantMsg]);
    setInput("");
    setIsLoading(true);

    try {
      console.log("[FRONTEND DEBUG] Making request to:", PROXY_URL);
      console.log("[FRONTEND DEBUG] Request payload:", {
        model: MODEL_NAME,
        messages: [...messages, userMsg].map(m => ({ role: m.role, content: m.content })),
        max_tokens: 10000,
        stream: true
      });
      
      const response = await fetch(PROXY_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: MODEL_NAME,
          messages: [...messages, userMsg].map(m => ({ role: m.role, content: m.content })),
          max_tokens: 10000,
          stream: true // Enable streaming
        })
      });

      console.log("[FRONTEND DEBUG] Response status:", response.status, "ok:", response.ok);
      if (!response.ok) {
        const errorText = await response.text();
        console.error("[FRONTEND DEBUG] Response error:", errorText);
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      if (!response.body) throw new Error("No response body");

      console.log("[FRONTEND DEBUG] Starting to read stream");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let streamDone = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          console.log("[FRONTEND DEBUG] Stream done, buffer remaining:", buffer.length);
          break;
        }

        const chunk = decoder.decode(value, { stream: true });
        console.log("[FRONTEND DEBUG] Received chunk:", chunk.substring(0, 100));
        buffer += chunk;
        
        // Process complete lines from buffer
        // Split on newlines and process each line
        const lines = buffer.split('\n');
        // Keep the last incomplete line in buffer
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          
          // Skip empty lines
          if (!trimmed) {
            console.log("[FRONTEND DEBUG] Skipping empty line");
            continue;
          }
          
          // Check for data: prefix
          if (!trimmed.startsWith("data: ")) {
            console.log("[FRONTEND DEBUG] Line doesn't start with 'data: ':", trimmed.substring(0, 50));
            continue;
          }
          
          const dataStr = trimmed.slice(6); // Remove "data: " prefix
          console.log("[FRONTEND DEBUG] Processing data string:", dataStr.substring(0, 100));
          
          // Check for [DONE] marker
          if (dataStr === "[DONE]") {
            console.log("[FRONTEND DEBUG] Received [DONE] marker");
            streamDone = true;
            break; // Break out of for loop
          }

          try {
            const data = JSON.parse(dataStr);
            console.log("[FRONTEND DEBUG] Parsed JSON, keys:", Object.keys(data));

            // --- CASE A: PROBE UPDATE (Custom Event) ---
            if (data.type === "probe_update") {
              console.log("[FRONTEND DEBUG] Processing probe_update");
              const deltaScores = data.probe_scores; // [n_probes, n_new_tokens]
              
              if (deltaScores && deltaScores.length > 0) {
                 console.log("[FRONTEND DEBUG] Probe update has", deltaScores.length, "probes,", deltaScores[0]?.length, "tokens");
                 setNumProbes(deltaScores.length);
                 
                 setMessages(prev => {
                   const newHistory = [...prev];
                   // Ensure we have an assistant message
                   if (newHistory.length === 0 || newHistory[newHistory.length - 1].role !== 'assistant') {
                     console.warn("[FRONTEND DEBUG] No assistant message found for probe update, creating one");
                     newHistory.push({
                       role: 'assistant',
                       content: '',
                       tokens: [],
                       scores: []
                     });
                   }
                   
                   const lastMsg = { ...newHistory[newHistory.length - 1] };
                   
                   // Initialize scores matrix if empty
                   if (!lastMsg.scores || lastMsg.scores.length === 0) {
                     lastMsg.scores = deltaScores;
                   } else {
                     // Append new columns to existing rows
                     lastMsg.scores = lastMsg.scores.map((row, probeIdx) => {
                       const newCols = deltaScores[probeIdx] || [];
                       return [...row, ...newCols];
                     });
                   }
                   
                   newHistory[newHistory.length - 1] = lastMsg;
                   return newHistory;
                 });
              }
              continue;
            }

            // --- CASE B: STANDARD TEXT TOKEN ---
            if (data.choices?.[0]?.delta?.content) {
              const tokenText = data.choices[0].delta.content;
              console.log("[FRONTEND DEBUG] Processing token text:", JSON.stringify(tokenText));
              
              setMessages(prev => {
                const newHistory = [...prev];
                // Make sure we have an assistant message to update
                if (newHistory.length === 0 || newHistory[newHistory.length - 1].role !== 'assistant') {
                  console.warn("[FRONTEND DEBUG] No assistant message found, creating one");
                  newHistory.push({
                    role: 'assistant',
                    content: '',
                    tokens: [],
                    scores: []
                  });
                }
                
                const lastMsg = { ...newHistory[newHistory.length - 1] };
                
                // Initialize if needed
                if (typeof lastMsg.content !== 'string') lastMsg.content = '';
                if (!Array.isArray(lastMsg.tokens)) lastMsg.tokens = [];
                
                lastMsg.content += tokenText;
                lastMsg.tokens = [...lastMsg.tokens, tokenText];
                
                console.log("[FRONTEND DEBUG] Updated message, total tokens:", lastMsg.tokens.length, "content length:", lastMsg.content.length);
                newHistory[newHistory.length - 1] = lastMsg;
                return newHistory;
              });
            } else {
              console.log("[FRONTEND DEBUG] No content in delta, choices:", data.choices);
            }

          } catch (err) {
            console.warn("[FRONTEND DEBUG] JSON Parse Error on chunk:", dataStr.substring(0, 100), err);
          }
        }
        
        // Break if we saw [DONE]
        if (streamDone) {
          console.log("[FRONTEND DEBUG] Breaking due to streamDone");
          break; // Break out of while loop
        }
      }
      
      // Process any remaining content in buffer after stream ends
      if (buffer && !streamDone) {
        const trimmed = buffer.trim();
        if (trimmed && trimmed.startsWith("data: ")) {
          const dataStr = trimmed.slice(6);
          if (dataStr !== "[DONE]") {
            try {
              const data = JSON.parse(dataStr);
              
              // Handle probe update
              if (data.type === "probe_update" && data.probe_scores) {
                const deltaScores = data.probe_scores;
                if (deltaScores && deltaScores.length > 0) {
                  setNumProbes(deltaScores.length);
                  setMessages(prev => {
                    const newHistory = [...prev];
                    const lastMsg = { ...newHistory[newHistory.length - 1] };
                    if (!lastMsg.scores || lastMsg.scores.length === 0) {
                      lastMsg.scores = deltaScores;
                    } else {
                      lastMsg.scores = lastMsg.scores.map((row, probeIdx) => {
                        const newCols = deltaScores[probeIdx] || [];
                        return [...row, ...newCols];
                      });
                    }
                    newHistory[newHistory.length - 1] = lastMsg;
                    return newHistory;
                  });
                }
              }
              
              // Handle text token
              if (data.choices?.[0]?.delta?.content) {
                const tokenText = data.choices[0].delta.content;
                setMessages(prev => {
                  const newHistory = [...prev];
                  const lastMsg = { ...newHistory[newHistory.length - 1] };
                  lastMsg.content += tokenText;
                  lastMsg.tokens = [...(lastMsg.tokens || []), tokenText];
                  newHistory[newHistory.length - 1] = lastMsg;
                  return newHistory;
                });
              }
            } catch (err) {
              console.warn("JSON Parse Error on final buffer:", err);
            }
          }
        }
      }

    } catch (err) {
      console.error(err);
      setMessages(prev => [...prev, { role: 'assistant', content: "\n[Connection Error]" }]);
    } finally {
      setIsLoading(false);
    }
  };

  // Prepare Graph Data for the LATEST assistant message only
  const lastMessage = messages.filter(m => m.role === 'assistant').slice(-1)[0];
  
  const graphData = useMemo(() => {
    if (!lastMessage || !lastMessage.scores || lastMessage.scores.length === 0) return [];
    
    // Transform from [n_probes, n_tokens] -> Array of { index, probe0, probe1... }
    // Note: Use tokens length or scores length (scores[0].length)
    const numTokens = lastMessage.scores[0].length;
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
    if (!probeRow || probeRow.length === 0) return 1;
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

        {messages.map((msg, i) => (
          <div key={i} className={cn("flex", msg.role === 'user' ? "justify-end" : "justify-start")}>
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
                  {/* If we have tokens with probe data, render with heatmap. Otherwise show plain content. */}
                  {msg.tokens && msg.tokens.length > 0 && msg.scores && msg.scores.length > 0 ? (
                    <div className="flex flex-wrap items-baseline content-start gap-y-1">
                      {msg.tokens.map((token, tIdx) => (
                        <TokenRenderer
                          key={tIdx}
                          text={token}
                          index={tIdx}
                          score={msg.scores?.[selectedProbe]?.[tIdx] || 0}
                          maxScore={currentMaxScore}
                          hoveredIndex={hoveredTokenIndex}
                          setHoveredIndex={setHoveredTokenIndex}
                        />
                      ))}
                    </div>
                  ) : (
                    msg.content ? (
                      <span>{msg.content}</span>
                    ) : (
                      <span className="animate-pulse text-gray-400">Thinking...</span>
                    )
                  )}
                </div>
              )}
            </div>
          </div>
        ))}
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