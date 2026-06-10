/// FraudX Analyst - Chat Screen
/// ================================================

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../models/models.dart';
import '../services/api_service.dart';
import '../config/api_config.dart';
import '../providers/app_provider.dart';
import '../widgets/animated_bot.dart';
import '../services/tutorial_service.dart';

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});
  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final _controller = TextEditingController();
  final _scrollController = ScrollController();
  final List<ChatMessage> _messages = [];
  bool _isLoading = false;
  List<String> _suggestions = [
    'What is credit card fraud?',
    'Explain XGBoost model',
    'How does fraud detection work?',
    'What does SHAP mean?',
  ];

  @override
  void initState() {
    super.initState();
    _loadSuggestions();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<AppProvider>().addListener(_onProviderChanged);
    });
  }

  void _onProviderChanged() {
    if (!mounted) return;
    final provider = context.read<AppProvider>();
    if (provider.pendingChatQuestion != null) {
      final question = provider.pendingChatQuestion!;
      provider.clearPendingChatQuestion();
      Future.microtask(() => _sendMessage(question));
    }
  }

  Future<void> _loadSuggestions() async {
    try {
      final suggestions = await ApiService.getChatSuggestions();
      if (mounted) setState(() => _suggestions = suggestions);
    } catch (_) {}
  }

  Future<void> _sendMessage(String text) async {
    if (text.trim().isEmpty) return;
    setState(() {
      _messages.add(ChatMessage(message: text, isUser: true, timestamp: DateTime.now()));
      _isLoading = true;
    });
    _controller.clear();
    _scrollToBottom();
    // Tell tutorial the chatbot has responded
        try {
          final tutorial = context.read<TutorialService>();
          if (tutorial.isActive && tutorial.waitingForAction) {
            tutorial.completeAction();
          }
        } catch (_) {}

    final prevMessages = _messages.length > 1 ? _messages.sublist(0, _messages.length - 1) : <ChatMessage>[];
    final history = prevMessages
        .reversed.take(6).toList().reversed
        .map((m) => {'role': m.isUser ? 'user' : 'assistant', 'content': m.message})
        .toList();

    final provider = context.read<AppProvider>();
    try {
      final response = await ApiService.chat(ChatRequest(
        message: text,
        deviceId: ApiConfig.deviceId,
        simulationId: provider.chatSimulationId,
        chatHistory: history.isNotEmpty ? history : null,
      ));
      if (mounted) {
        setState(() {
          _messages.add(ChatMessage(message: response.reply, isUser: false, timestamp: DateTime.now(), sources: response.sources));
          _isLoading = false;
        });
        _scrollToBottom();
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _messages.add(ChatMessage(message: 'Sorry, I encountered an error. Please try again.', isUser: false, timestamp: DateTime.now()));
          _isLoading = false;
        });
        _scrollToBottom();
      }
    }
  }

  void _askAboutLastSimulation() {
    final provider = context.read<AppProvider>();
    final pred = provider.lastPrediction;
    if (pred != null) {
      provider.clearChatContext();
      _sendMessage(
        'Explain my last simulation result (ID: ${pred.simulationId}). '
        'The prediction was ${pred.prediction} with a risk score of ${(pred.riskScore * 100).toStringAsFixed(1)}%. '
        'Why was this transaction classified this way? What features contributed most?'
      );
    } else {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
        content: Text('No simulation yet. Run a simulation first!'),
        backgroundColor: Color(0xFFFF9800),
      ));
    }
  }

  void _scrollToBottom() {
    // Scroll twice — first quickly, then after render to catch final position
    Future.delayed(const Duration(milliseconds: 100), () {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(_scrollController.position.maxScrollExtent, duration: const Duration(milliseconds: 300), curve: Curves.easeOut);
      }
    });
    Future.delayed(const Duration(milliseconds: 500), () {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(_scrollController.position.maxScrollExtent, duration: const Duration(milliseconds: 200), curve: Curves.easeOut);
      }
    });
  }

  @override
  void dispose() {
    try { context.read<AppProvider>().removeListener(_onProviderChanged); } catch (_) {}
    _controller.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final hasSimulation = context.watch<AppProvider>().lastPrediction != null;
    final hasMessages = _messages.isNotEmpty;

    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: Column(children: [
          // ── Minimal Header ───────────────────────────────────
          Container(
            padding: const EdgeInsets.fromLTRB(20, 10, 20, 10),
            child: Row(children: [
              AnimatedBot(size: 36, isActive: true),
              const SizedBox(width: 12),
              const Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text('FraudX Assistant', style: TextStyle(fontSize: 17, fontWeight: FontWeight.w700, color: Color(0xFF1A1A2E))),
                Text('Powered by RAG + Gemini', style: TextStyle(fontSize: 11, color: Color(0xFF9CA3AF), fontWeight: FontWeight.w500)),
              ]),
              const Spacer(),
              if (_messages.isNotEmpty)
                GestureDetector(
                  onTap: () => setState(() { _messages.clear(); }),
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    decoration: BoxDecoration(color: const Color(0xFFF3F4F6), borderRadius: BorderRadius.circular(20)),
                    child: const Row(mainAxisSize: MainAxisSize.min, children: [
                      Icon(Icons.add, size: 16, color: Color(0xFF6B7280)),
                      SizedBox(width: 4),
                      Text('New', style: TextStyle(fontSize: 13, fontWeight: FontWeight.w500, color: Color(0xFF6B7280))),
                    ]),
                  ),
                ),
            ]),
          ),
          const Divider(height: 1, color: Color(0xFFF3F4F6)),

          // ── Messages or Welcome ──────────────────────────────
          Expanded(
            child: !hasMessages
                ? _WelcomeView(
                    suggestions: _suggestions,
                    onSuggestionTap: _sendMessage,
                    hasSimulation: hasSimulation,
                    onAskSimulation: _askAboutLastSimulation,
                  )
                : ListView.builder(
                    controller: _scrollController,
                    padding: const EdgeInsets.fromLTRB(0, 12, 0, 8),
                    itemCount: _messages.length + (_isLoading ? 1 : 0),
                    itemBuilder: (ctx, index) {
                      if (index == _messages.length && _isLoading) return const _TypingIndicator();
                      return _GeminiMessage(message: _messages[index]);
                    },
                  ),
          ),

          // ── Suggestion chips (after first few messages) ─────
          if (hasMessages && _messages.length <= 3)
            SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              padding: const EdgeInsets.fromLTRB(20, 4, 20, 4),
              child: Row(
                children: _suggestions.take(4).map((s) => Padding(
                  padding: const EdgeInsets.only(right: 8),
                  child: GestureDetector(
                    onTap: () => _sendMessage(s),
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                      decoration: BoxDecoration(
                        border: Border.all(color: const Color(0xFFE5E7EB)),
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Text(s, style: const TextStyle(fontSize: 13, color: Color(0xFF374151))),
                    ),
                  ),
                )).toList(),
              ),
            ),

          // ── Input Bar (Gemini-style pill) ────────────────────
          Container(
            margin: const EdgeInsets.fromLTRB(16, 8, 16, 12),
            decoration: BoxDecoration(
              color: const Color(0xFFF5F7FA),
              borderRadius: BorderRadius.circular(28),
              border: Border.all(color: const Color(0xFFE5E7EB)),
            ),
            child: Row(children: [
              const SizedBox(width: 20),
              Expanded(
                child: TextField(
                  controller: _controller,
                  onSubmitted: _sendMessage,
                  style: const TextStyle(fontSize: 16, color: Color(0xFF1A1A2E)),
                  decoration: const InputDecoration(
                    hintText: 'Ask about fraud detection…',
                    hintStyle: TextStyle(fontSize: 15, color: Color(0xFFADB5BD)),
                    border: InputBorder.none,
                    contentPadding: EdgeInsets.symmetric(vertical: 14),
                  ),
                ),
              ),
              GestureDetector(
                onTap: () => _sendMessage(_controller.text),
                child: Container(
                  width: 40, height: 40,
                  margin: const EdgeInsets.only(right: 6),
                  decoration: const BoxDecoration(
                    color: Color(0xFF2A9D8F),
                    shape: BoxShape.circle,
                  ),
                  child: const Icon(Icons.arrow_upward, size: 20, color: Colors.white),
                ),
              ),
            ]),
          ),
        ]),
      ),
    );
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  Welcome View (shown when no messages — like Gemini's start screen)
// ══════════════════════════════════════════════════════════════════════════════

class _WelcomeView extends StatelessWidget {
  final List<String> suggestions;
  final Function(String) onSuggestionTap;
  final bool hasSimulation;
  final VoidCallback onAskSimulation;

  const _WelcomeView({
    required this.suggestions,
    required this.onSuggestionTap,
    required this.hasSimulation,
    required this.onAskSimulation,
  });

  @override
  Widget build(BuildContext context) {
    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 32),
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          AnimatedBot(size: 72, isActive: true),
          const SizedBox(height: 20),
          const Text('FraudX Assistant', style: TextStyle(fontSize: 24, fontWeight: FontWeight.w800, color: Color(0xFF1A1A2E))),
          const SizedBox(height: 8),
          const Text(
            'Your AI fraud detection expert.\nAsk me anything about credit card security.',
            textAlign: TextAlign.center,
            style: TextStyle(fontSize: 15, height: 1.5, color: Color(0xFF9CA3AF)),
          ),
          const SizedBox(height: 32),
          ...suggestions.take(4).map((s) => Padding(
            padding: const EdgeInsets.only(bottom: 10),
            child: GestureDetector(
              onTap: () => onSuggestionTap(s),
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: const Color(0xFFE5E7EB)),
                ),
                child: Row(children: [
                  const Icon(Icons.auto_awesome, size: 18, color: Color(0xFF2A9D8F)),
                  const SizedBox(width: 12),
                  Expanded(child: Text(s, style: const TextStyle(fontSize: 15, color: Color(0xFF374151)))),
                  const Icon(Icons.arrow_forward_ios, size: 14, color: Color(0xFFD1D5DB)),
                ]),
              ),
            ),
          )),
          if (hasSimulation) ...[
            const SizedBox(height: 8),
            GestureDetector(
              onTap: onAskSimulation,
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
                decoration: BoxDecoration(
                  color: const Color(0xFFE0F2F1),
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: const Color(0xFF2A9D8F).withOpacity(0.3)),
                ),
                child: const Row(children: [
                  Icon(Icons.science, size: 18, color: Color(0xFF2A9D8F)),
                  SizedBox(width: 12),
                  Expanded(child: Text('Explain my last simulation', style: TextStyle(fontSize: 15, fontWeight: FontWeight.w600, color: Color(0xFF2A9D8F)))),
                  Icon(Icons.arrow_forward_ios, size: 14, color: Color(0xFF2A9D8F)),
                ]),
              ),
            ),
          ],
        ]),
      ),
    );
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  Gemini-style Message
// ══════════════════════════════════════════════════════════════════════════════

class _GeminiMessage extends StatelessWidget {
  final ChatMessage message;
  const _GeminiMessage({required this.message});

  @override
  Widget build(BuildContext context) {
    final isUser = message.isUser;

    if (isUser) {
      return Padding(
        padding: const EdgeInsets.fromLTRB(60, 4, 20, 4),
        child: Align(
          alignment: Alignment.centerRight,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 12),
            decoration: BoxDecoration(
              color: const Color(0xFF2A9D8F),
              borderRadius: BorderRadius.circular(22),
            ),
            child: Text(message.message, style: const TextStyle(fontSize: 15, height: 1.5, color: Colors.white)),
          ),
        ),
      );
    }

    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 8, 40, 8),
      child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Padding(
          padding: const EdgeInsets.only(top: 2),
          child: AnimatedBot(size: 28, isActive: true),
        ),
        const SizedBox(width: 12),
        Flexible(
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            RichText(text: _parseMarkdown(message.message, const Color(0xFF1A1A2E))),
            const SizedBox(height: 6),
            Row(children: [
              Text(_formatTime(message.timestamp), style: const TextStyle(fontSize: 11, color: Color(0xFFADB5BD))),
              if (message.sources != null && message.sources!.isNotEmpty) ...[
                const SizedBox(width: 8),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                  decoration: BoxDecoration(color: const Color(0xFFF0FDF9), borderRadius: BorderRadius.circular(8)),
                  child: const Row(mainAxisSize: MainAxisSize.min, children: [
                    Icon(Icons.menu_book, size: 11, color: Color(0xFF2A9D8F)),
                    SizedBox(width: 3),
                    Text('Knowledge Base', style: TextStyle(fontSize: 10, color: Color(0xFF2A9D8F), fontWeight: FontWeight.w500)),
                  ]),
                ),
              ],
            ]),
          ]),
        ),
      ]),
    );
  }

  String _formatTime(DateTime dt) {
    final diff = DateTime.now().difference(dt);
    if (diff.inSeconds < 60) return 'Just now';
    if (diff.inMinutes < 60) return '${diff.inMinutes}m ago';
    return '${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}';
  }

  /// Parses **bold** and *italic* markdown into styled TextSpan
  TextSpan _parseMarkdown(String text, Color textColor) {
    // Clean stray asterisks that are missed sometimes
    text = text.split('\n').map((line) => line.replaceAll(RegExp(r'^\s+'), '')).join('\n'); //trim each line first
    text = text.replaceAll(RegExp(r'(?<!\*)\*(?!\*)'), '');
    final spans = <TextSpan>[];
    final regex = RegExp(r'\*\*(.+?)\*\*|\*(.+?)\*(?=[^*])');
    int lastEnd = 0;

    for (final match in regex.allMatches(text)) {
      // Add text before the match
      if (match.start > lastEnd) {
        spans.add(TextSpan(text: text.substring(lastEnd, match.start)));
      }

      if (match.group(1) != null) {
        // **bold**
        spans.add(TextSpan(
          text: match.group(1),
          style: const TextStyle(fontWeight: FontWeight.w700),
        ));
      } else if (match.group(2) != null) {
        // *italic*
        spans.add(TextSpan(
          text: match.group(2),
          style: const TextStyle(fontStyle: FontStyle.italic),
        ));
      }
      lastEnd = match.end;
    }

    // Add remaining text
    if (lastEnd < text.length) {
      spans.add(TextSpan(text: text.substring(lastEnd)));
    }

    return TextSpan(
      style: TextStyle(fontSize: 15, height: 1.6, color: textColor),
      children: spans.isEmpty ? [TextSpan(text: text)] : spans,
    );
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  Typing Indicator (animated dots)
// ══════════════════════════════════════════════════════════════════════════════

class _TypingIndicator extends StatefulWidget {
  const _TypingIndicator();
  @override
  State<_TypingIndicator> createState() => _TypingIndicatorState();
}

class _TypingIndicatorState extends State<_TypingIndicator> with SingleTickerProviderStateMixin {
  late AnimationController _anim;

  @override
  void initState() {
    super.initState();
    _anim = AnimationController(vsync: this, duration: const Duration(milliseconds: 1200))..repeat();
  }

  @override
  void dispose() { _anim.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 8, 40, 8),
      child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
        AnimatedBot(size: 28, isActive: true),
        const SizedBox(width: 12),
        AnimatedBuilder(
          animation: _anim,
          builder: (context, _) {
            return Row(mainAxisSize: MainAxisSize.min, children: List.generate(3, (i) {
              final delay = i * 0.2;
              final value = ((_anim.value + delay) % 1.0);
              final opacity = (value < 0.5) ? 0.3 + value * 1.0 : 0.3 + (1.0 - value) * 1.0;
              return Container(
                width: 8, height: 8,
                margin: const EdgeInsets.symmetric(horizontal: 3),
                decoration: BoxDecoration(
                  color: const Color(0xFF2A9D8F).withOpacity(opacity.clamp(0.3, 1.0)),
                  shape: BoxShape.circle,
                ),
              );
            }));
          },
        ),
      ]),
    );
  }
}
