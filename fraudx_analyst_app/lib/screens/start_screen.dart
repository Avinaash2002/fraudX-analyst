/// FraudX Analyst - Start Screen
/// ================================
/// Colors: teal-green (#2A9D8F → #4ECDC4) matching Lovable design
/// 3 controllers: entrance, loop (scan+particles+dot), button (breathe+glow)

import 'dart:math';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

// ── Lovable gradient: green-teal + sky-blue + light-green ────────────────────
const _kGreenTeal = Color(0xFF2A9D8F);  // green-teal
const _kSkyBlue = Color(0xFF38BDF8);    // sky-blue accent
const _kLightGreen = Color(0xFF6BCB77); // light green
const _kTeal = Color(0xFF2A9D8F);       // base teal (icons, accents)
const _kBg = Color(0xFFF5F7FA);
const _kDark = Color(0xFF1A1A2E);

class StartScreen extends StatefulWidget {
  const StartScreen({super.key});
  @override
  State<StartScreen> createState() => _StartScreenState();
}

class _StartScreenState extends State<StartScreen> with TickerProviderStateMixin {
  // 1) Entrance (one-shot)
  late AnimationController _entrance;
  late Animation<double> _logoScale, _logoRotate, _fadeIn;

  // 2) Loop — scan line, particles, dot pulse (single controller)
  late AnimationController _loop;

  // 3) Button — breathing + glow sweep
  late AnimationController _button;
  late Animation<double> _btnScale;

  bool _isStarting = false;
  bool _isOnline = true; // toggle for demo purposes (simulate offline mode)

  @override
  void initState() {
    super.initState();

    _entrance = AnimationController(vsync: this, duration: const Duration(milliseconds: 1200));
    _logoScale = Tween(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _entrance, curve: const Interval(0.0, 0.6, curve: Curves.elasticOut)));
    _logoRotate = Tween(begin: -0.5, end: 0.0).animate(
      CurvedAnimation(parent: _entrance, curve: const Interval(0.0, 0.6, curve: Curves.elasticOut)));
    _fadeIn = Tween(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _entrance, curve: const Interval(0.2, 0.7, curve: Curves.easeOut)));
    _entrance.forward();
    _checkConnectivity();

    _loop = AnimationController(vsync: this, duration: const Duration(seconds: 5))..repeat();

    _button = AnimationController(vsync: this, duration: const Duration(seconds: 2))..repeat();
    _btnScale = TweenSequence<double>([
      TweenSequenceItem(tween: Tween(begin: 1.0, end: 1.045), weight: 50),
      TweenSequenceItem(tween: Tween(begin: 1.045, end: 1.0), weight: 50),
    ]).animate(CurvedAnimation(parent: _button, curve: Curves.easeInOut));
  }

  @override
  void dispose() {
    _entrance.dispose();
    _loop.dispose();
    _button.dispose();
    super.dispose();
  }

  Future<void> _start() async {
    setState(() => _isStarting = true);
    await Future.delayed(const Duration(milliseconds: 1200));
    if (!mounted) return;
    Navigator.of(context).pushReplacementNamed('/main');
  }

  Future<void> _checkConnectivity() async {
    try {
      final response = await http.get(Uri.parse('https://www.google.com')).timeout(const Duration(seconds: 5));
      if (mounted) setState(() => _isOnline = response.statusCode == 200);
    } catch (_) {
      if (mounted) setState(() => _isOnline = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _kBg,
      body: Stack(
        children: [
          // Grid background
          Positioned.fill(child: Opacity(opacity: 0.03, child: CustomPaint(painter: _GridPainter()))),

          // Floating particles
          ...List.generate(6, (i) => AnimatedBuilder(
            animation: _loop,
            builder: (_, __) => _buildParticle(i),
          )),

          // Main content
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 28),
              child: Column(children: [
                const Spacer(flex: 2),

                // Logo (spring)
                AnimatedBuilder(
                  animation: _entrance,
                  builder: (_, child) => Transform.scale(
                    scale: _logoScale.value,
                    child: Transform.rotate(angle: _logoRotate.value * pi, child: child),
                  ),
                  child: _buildLogo(),
                ),
                const SizedBox(height: 24),

                // Title
                FadeTransition(opacity: _fadeIn, child: _buildTitle()),
                const SizedBox(height: 28),

                // Feature cards
                FadeTransition(opacity: _fadeIn, child: _buildCards()),
                const Spacer(flex: 2),

                // About section
                FadeTransition(
                  opacity: _fadeIn,
                  child: Container(
                    margin: const EdgeInsets.symmetric(horizontal: 8),
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.7),
                      borderRadius: BorderRadius.circular(14),
                      border: Border.all(color: _kGreenTeal.withOpacity(0.15)),
                    ),
                    child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                      Icon(Icons.info_outline, size: 18, color: _kGreenTeal.withOpacity(0.7)),
                      const SizedBox(width: 10),
                      const Expanded(child: Text(
                        'FraudX Analyst is an AI-powered credit card fraud detection system that uses machine learning models (XGBoost, LightGBM, Autoencoder) with SHAP explainability and a RAG chatbot to detect, analyse, and explain potentially fraudulent transactions in real time.',
                        style: TextStyle(fontSize: 12, height: 1.5, color: Color(0xFF6B7280)),
                      )),
                    ]),
                  ),
                ),
                const SizedBox(height: 16),

                // Start button (breathing)
                FadeTransition(
                  opacity: _fadeIn,
                  child: AnimatedBuilder(
                    animation: _button,
                    builder: (_, child) => Transform.scale(
                      scale: _isStarting ? 1.0 : _btnScale.value,
                      child: child,
                    ),
                    child: SizedBox(width: double.infinity, height: 52, child: _buildButton()),
                  ),
                ),
                const SizedBox(height: 24),

                // Status bar
                FadeTransition(opacity: _fadeIn, child: _buildStatusBar()),
                const SizedBox(height: 16),
              ]),
            ),
          ),
        ],
      ),
    );
  }

  // ════════════════════════════════════════════════════════════════════════════
  //  LOGO — teal gradient, scanning line, corner brackets
  // ════════════════════════════════════════════════════════════════════════════

  Widget _buildLogo() {
    return SizedBox(
      width: 88, height: 88,
      child: Stack(
        clipBehavior: Clip.none,
        children: [
          // Outer glow
          Positioned(left: -10, top: -10, right: -10, bottom: -10,
            child: Container(decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(28),
              boxShadow: [
                BoxShadow(color: _kGreenTeal.withOpacity(0.2), blurRadius: 28, spreadRadius: 2),
                BoxShadow(color: _kSkyBlue.withOpacity(0.12), blurRadius: 24, spreadRadius: 1),
                BoxShadow(color: _kLightGreen.withOpacity(0.1), blurRadius: 40, spreadRadius: 4),
              ],
            )),
          ),

          // Main container (gradient + shield icon ONLY — no scan line here)
          Positioned.fill(
            child: ClipRRect(
              borderRadius: BorderRadius.circular(20),
              child: Stack(children: [
                // Green-teal → sky-blue → light-green gradient (Lovable mix)
                Container(decoration: const BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.topLeft, end: Alignment.bottomRight,
                    colors: [_kGreenTeal, _kSkyBlue, _kLightGreen],
                  ),
                )),
                // Shield icon
                const Center(child: Icon(Icons.shield_outlined, size: 42, color: Colors.white)),
              ]),
            ),
          ),

          // Scanning line — OUTSIDE ClipRRect, extends beyond shield
          Positioned(
            left: -20, right: -20,
            top: 0, bottom: 0,
            child: AnimatedBuilder(
              animation: _loop,
              builder: (_, __) {
                // Smooth sine bounce: continuous up↔down with no jerk
                final t = (sin(_loop.value * 2 * pi * 0.7) + 1) / 2; // 0→1 smooth
                final yPos = t * 76; // range of travel (leave margin top/bottom)
                return Stack(children: [
                  Positioned(
                    left: 0, right: 0,
                    top: yPos,
                    child: Container(
                      height: 4, // the line thickness (scanning bar)
                      decoration: BoxDecoration(
                        gradient: LinearGradient(colors: [
                          Colors.transparent,
                          Colors.transparent,
                          Colors.white.withOpacity(0.3),
                          Colors.white.withOpacity(0.7),
                          Colors.white.withOpacity(0.9),
                          Colors.white.withOpacity(0.7),
                          Colors.white.withOpacity(0.3),
                          Colors.transparent,
                          Colors.transparent,
                        ]),
                        boxShadow: [
                          BoxShadow(color: Colors.white.withOpacity(0.6), blurRadius: 12, spreadRadius: 2),
                          BoxShadow(color: _kSkyBlue.withOpacity(0.3), blurRadius: 20, spreadRadius: 4),
                        ],
                      ),
                    ),
                  ),
                ]);
              },
            ),
          ),

          // Corner brackets (green-teal top, light-green bottom)
          _corner(top: -4, left: -4, bTop: true, bLeft: true, color: _kGreenTeal),
          _corner(top: -4, right: -4, bTop: true, bRight: true, color: _kGreenTeal),
          _corner(bottom: -4, left: -4, bBottom: true, bLeft: true, color: _kLightGreen),
          _corner(bottom: -4, right: -4, bBottom: true, bRight: true, color: _kLightGreen),
        ],
      ),
    );
  }

  Widget _corner({
    double? top, double? bottom, double? left, double? right,
    bool bTop = false, bool bBottom = false,
    bool bLeft = false, bool bRight = false,
    required Color color,
  }) {
    return Positioned(
      top: top, bottom: bottom, left: left, right: right,
      child: Container(
        width: 14, height: 14,
        decoration: BoxDecoration(
          border: Border(
            top: bTop ? BorderSide(width: 2, color: color) : BorderSide.none,
            bottom: bBottom ? BorderSide(width: 2, color: color) : BorderSide.none,
            left: bLeft ? BorderSide(width: 2, color: color) : BorderSide.none,
            right: bRight ? BorderSide(width: 2, color: color) : BorderSide.none,
          ),
        ),
      ),
    );
  }

  // ════════════════════════════════════════════════════════════════════════════
  //  TITLE
  // ════════════════════════════════════════════════════════════════════════════

  Widget _buildTitle() {
    return Column(children: [
      RichText(text: const TextSpan(
        style: TextStyle(fontSize: 26, fontWeight: FontWeight.w800, color: _kDark, letterSpacing: -0.5),
        children: [
          TextSpan(text: 'Fraud'),
          TextSpan(text: 'X', style: TextStyle(color: _kTeal)),
          TextSpan(text: '-Analyst'),
        ],
      )),
      const SizedBox(height: 6),
      // v1.0.0 with decorative lines
      Row(mainAxisAlignment: MainAxisAlignment.center, children: [
        Container(width: 32, height: 1, decoration: BoxDecoration(
          gradient: LinearGradient(colors: [Colors.transparent, _kTeal.withOpacity(0.5)]))),
        const SizedBox(width: 8),
        Text('v1.0.0', style: TextStyle(fontSize: 11, fontWeight: FontWeight.w500, fontFamily: 'monospace', color: Colors.grey.shade500, letterSpacing: 1.5)),
        const SizedBox(width: 8),
        Container(width: 32, height: 1, decoration: BoxDecoration(
          gradient: LinearGradient(colors: [_kTeal.withOpacity(0.5), Colors.transparent]))),
      ]),
      const SizedBox(height: 10),
      Text('AI-Powered Fraud Detection &\nTransaction Security', textAlign: TextAlign.center,
        style: TextStyle(fontSize: 13, color: Colors.grey.shade500, height: 1.5)),
    ]);
  }

  // ════════════════════════════════════════════════════════════════════════════
  //  FEATURE CARDS
  // ════════════════════════════════════════════════════════════════════════════

  Widget _buildCards() {
    return Column(children: [
      Row(children: [
        Expanded(child: _card(Icons.monitor_heart_outlined, 'Real-time', 'Monitoring', 0)),
        const SizedBox(width: 10),
        Expanded(child: _card(Icons.memory, 'ML-Based', 'Detection', 1)),
      ]),
      const SizedBox(height: 10),
      Row(children: [
        Expanded(child: _card(Icons.lock_outlined, 'Secure', 'Processing', 2)),
        const SizedBox(width: 10),
        Expanded(child: _card(Icons.bolt_outlined, 'Fast', 'Analysis', 3)),
      ]),
    ]);
  }

  Widget _card(IconData icon, String title, String sub, int i) {
    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0.0, end: 1.0),
      duration: Duration(milliseconds: 600 + i * 80),
      curve: Curves.easeOut,
      builder: (_, v, child) => Opacity(opacity: v, child: Transform.scale(scale: 0.9 + 0.1 * v, child: child)),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.7),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: Colors.grey.shade200.withOpacity(0.5)),
        ),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Icon(icon, size: 22, color: _kTeal),
          const SizedBox(height: 8),
          Text(title, style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w700, color: _kDark)),
          Text(sub, style: TextStyle(fontSize: 12, color: Colors.grey.shade500)),
        ]),
      ),
    );
  }

  // ════════════════════════════════════════════════════════════════════════════
  //  BUTTON — teal gradient, glow sweep, breathing
  // ════════════════════════════════════════════════════════════════════════════

  Widget _buildButton() {
    return GestureDetector(
      onTap: _isStarting ? null : _start,
      child: AnimatedBuilder(
        animation: _button,
        builder: (_, __) {
          final sweepPos = (_button.value * 2 - 0.5); // -0.5 → 1.5
          return Container(
            decoration: BoxDecoration(
              gradient: const LinearGradient(colors: [_kGreenTeal, _kSkyBlue, _kLightGreen]),
              borderRadius: BorderRadius.circular(14),
              boxShadow: [BoxShadow(color: _kTeal.withOpacity(0.3), blurRadius: 16, offset: const Offset(0, 6))],
            ),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(14),
              child: Stack(children: [
                // Glow sweep
                Positioned.fill(child: CustomPaint(painter: _GlowPainter(sweepPos))),
                // Text
                Center(child: _isStarting
                    ? const Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                        SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2.5, color: Colors.white)),
                        SizedBox(width: 12),
                        Text('Starting…', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: Colors.white)),
                      ])
                    : const Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                        Text('Start', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: Colors.white, letterSpacing: 0.3)),
                        SizedBox(width: 6),
                        Icon(Icons.chevron_right, size: 20, color: Colors.white),
                      ]),
                ),
              ]),
            ),
          );
        },
      ),
    );
  }

  // ════════════════════════════════════════════════════════════════════════════
  //  STATUS BAR
  // ════════════════════════════════════════════════════════════════════════════

  Widget _buildStatusBar() {
    return AnimatedBuilder(
      animation: _loop,
      builder: (_, __) {
        final dotOpacity = 0.4 + 0.6 * ((sin(_loop.value * 2 * pi * 2) + 1) / 2);
        return Row(mainAxisAlignment: MainAxisAlignment.center, children: [
          Opacity(
            opacity: dotOpacity,
            child: Container(width: 6, height: 6, decoration: BoxDecoration(
              color: _isOnline ? const Color(0xFF4CAF50) : const Color(0xFFEF4444),
              shape: BoxShape.circle,
            )),
          ),
          const SizedBox(width: 6),
          Text(
            _isOnline ? 'SYSTEMS ONLINE' : 'SYSTEMS OFFLINE',
            style: TextStyle(fontSize: 10, fontWeight: FontWeight.w600, fontFamily: 'monospace',
              color: _isOnline ? Colors.grey.shade500 : const Color(0xFFEF4444), letterSpacing: 0.5),
          ),
          Container(width: 1, height: 12, margin: const EdgeInsets.symmetric(horizontal: 12), color: Colors.grey.shade300),
          Text('3 MODELS READY', style: TextStyle(fontSize: 10, fontWeight: FontWeight.w600, fontFamily: 'monospace', color: Colors.grey.shade500, letterSpacing: 0.5)),
        ]);
      },
    );
  }

  // ════════════════════════════════════════════════════════════════════════════
  //  PARTICLES
  // ════════════════════════════════════════════════════════════════════════════

  Widget _buildParticle(int i) {
    final rng = Random(i * 42);
    final startX = rng.nextDouble() * 300 + 40;
    final startY = rng.nextDouble() * 500 + 100;
    final drift = rng.nextDouble() * 60 + 20;
    final phase = (_loop.value + i * 0.17) % 1.0;
    double opacity;
    if (phase < 0.3) {
      opacity = phase / 0.3 * 0.5;
    } else if (phase < 0.7) {
      opacity = 0.5;
    } else {
      opacity = (1.0 - phase) / 0.3 * 0.5;
    }
    return Positioned(
      left: startX, top: startY - drift * phase,
      child: Opacity(
        opacity: opacity.clamp(0.0, 0.5),
        child: Container(width: 4, height: 4, decoration: BoxDecoration(color: _kTeal.withOpacity(0.4), shape: BoxShape.circle)),
      ),
    );
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  PAINTERS
// ══════════════════════════════════════════════════════════════════════════════

class _GridPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()..color = _kTeal..strokeWidth = 0.5;
    for (double x = 0; x < size.width; x += 20) {
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), paint);
    }
    for (double y = 0; y < size.height; y += 20) {
      canvas.drawLine(Offset(0, y), Offset(size.width, y), paint);
    }
  }

  @override
  bool shouldRepaint(_) => false;
}

class _GlowPainter extends CustomPainter {
  final double position;
  _GlowPainter(this.position);

  @override
  void paint(Canvas canvas, Size size) {
    final center = size.width * position;
    final w = size.width * 0.35;
    final rect = Rect.fromLTWH(center - w / 2, 0, w, size.height);
    final paint = Paint()
      ..shader = LinearGradient(colors: [
        Colors.transparent,
        Colors.white.withOpacity(0.12),
        Colors.transparent,
      ]).createShader(rect);
    canvas.drawRect(Rect.fromLTWH(0, 0, size.width, size.height), paint);
  }

  @override
  bool shouldRepaint(_GlowPainter old) => (old.position - position).abs() > 0.01;
}
