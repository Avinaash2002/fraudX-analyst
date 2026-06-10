/// FraudX Analyst - Animated Bot Widget
/// ========================================
/// Reusable robot face with blinking eyes and looking around animation.
/// Used in chat screen header and floating action button.

import 'package:flutter/material.dart';
import 'dart:math';

class AnimatedBot extends StatefulWidget {
  final double size;
  final bool isActive; // true = eyes move & blink, false = static

  const AnimatedBot({super.key, this.size = 44, this.isActive = true});

  @override
  State<AnimatedBot> createState() => _AnimatedBotState();
}

class _AnimatedBotState extends State<AnimatedBot> with TickerProviderStateMixin {
  late AnimationController _blinkController;
  late AnimationController _lookController;
  late Animation<double> _blinkAnimation;
  late Animation<double> _lookXAnimation;
  late Animation<double> _lookYAnimation;
  final Random _random = Random();

  @override
  void initState() {
    super.initState();

    // Blink animation
    _blinkController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 150),
    );
    _blinkAnimation = Tween<double>(begin: 1.0, end: 0.1).animate(
      CurvedAnimation(parent: _blinkController, curve: Curves.easeInOut),
    );
    _blinkController.addStatusListener((status) {
      if (status == AnimationStatus.completed) {
        _blinkController.reverse();
      }
    });

    // Look around animation
    _lookController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 800),
    );
    _lookXAnimation = Tween<double>(begin: 0, end: 0).animate(
      CurvedAnimation(parent: _lookController, curve: Curves.easeInOut),
    );
    _lookYAnimation = Tween<double>(begin: 0, end: 0).animate(
      CurvedAnimation(parent: _lookController, curve: Curves.easeInOut),
    );

    if (widget.isActive) {
      _startBlinking();
      _startLooking();
    }
  }

  bool _firstBlink = true;
  bool _firstLook = true;

  void _startBlinking() {
    final delay = _firstBlink ? 400 : 1500 + _random.nextInt(3000);
    _firstBlink = false;
    Future.delayed(Duration(milliseconds: delay), () {
      if (mounted && widget.isActive) {
        _blinkController.forward();
        _startBlinking();
      }
    });
  }

  void _startLooking() {
    final delay = _firstLook ? 300 : 2000 + _random.nextInt(2500);
    _firstLook = false;
    Future.delayed(Duration(milliseconds: delay), () {
      if (mounted && widget.isActive) {
        final newX = (_random.nextDouble() - 0.5) * 2.5;
        final newY = (_random.nextDouble() - 0.5) * 1.5;
        _lookXAnimation = Tween<double>(begin: _lookXAnimation.value, end: newX).animate(
          CurvedAnimation(parent: _lookController, curve: Curves.easeInOut),
        );
        _lookYAnimation = Tween<double>(begin: _lookYAnimation.value, end: newY).animate(
          CurvedAnimation(parent: _lookController, curve: Curves.easeInOut),
        );
        _lookController.forward(from: 0);
        _startLooking();
      }
    });
  }

  @override
  void dispose() {
    _blinkController.dispose();
    _lookController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final s = widget.size;
    return SizedBox(
      width: s,
      height: s,
      child: AnimatedBuilder(
        animation: Listenable.merge([_blinkController, _lookController]),
        builder: (context, _) {
          return CustomPaint(
            painter: _BotPainter(
              blinkValue: _blinkAnimation.value,
              lookX: _lookXAnimation.value,
              lookY: _lookYAnimation.value,
            ),
            size: Size(s, s),
          );
        },
      ),
    );
  }
}

class _BotPainter extends CustomPainter {
  final double blinkValue;
  final double lookX;
  final double lookY;

  _BotPainter({
    required this.blinkValue,
    required this.lookX,
    required this.lookY,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final s = size.width;
    final center = Offset(s / 2, s / 2);

    // ── Head (rounded square) ──────────────────────
    final headPaint = Paint()..color = const Color(0xFF2A9D8F);
    final headRect = RRect.fromRectAndRadius(
      Rect.fromCenter(center: center, width: s * 0.82, height: s * 0.72),
      Radius.circular(s * 0.18),
    );
    canvas.drawRRect(headRect, headPaint);

    // ── Antenna ────────────────────────────────────
    final antennaPaint = Paint()
      ..color = const Color(0xFF2A9D8F)
      ..strokeWidth = s * 0.05
      ..strokeCap = StrokeCap.round;
    canvas.drawLine(
      Offset(s * 0.5, s * 0.15),
      Offset(s * 0.5, s * 0.05),
      antennaPaint,
    );
    final antennaDotPaint = Paint()..color = const Color(0xFF4ECDC4);
    canvas.drawCircle(Offset(s * 0.5, s * 0.04), s * 0.04, antennaDotPaint);

    // ── Eye sockets (white) ────────────────────────
    final socketPaint = Paint()..color = Colors.white;
    final eyeW = s * 0.2;
    final eyeH = s * 0.18 * blinkValue;
    final leftEyeCenter = Offset(s * 0.34, s * 0.46);
    final rightEyeCenter = Offset(s * 0.66, s * 0.46);

    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromCenter(center: leftEyeCenter, width: eyeW, height: eyeH),
        Radius.circular(s * 0.06),
      ),
      socketPaint,
    );
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromCenter(center: rightEyeCenter, width: eyeW, height: eyeH),
        Radius.circular(s * 0.06),
      ),
      socketPaint,
    );

    // ── Pupils (move with lookX/lookY) ─────────────
    if (blinkValue > 0.3) {
      final pupilPaint = Paint()..color = const Color(0xFF1A1A2E);
      final pupilR = s * 0.05;
      final offsetX = lookX * s * 0.03;
      final offsetY = lookY * s * 0.02;

      canvas.drawCircle(
        Offset(leftEyeCenter.dx + offsetX, leftEyeCenter.dy + offsetY),
        pupilR,
        pupilPaint,
      );
      canvas.drawCircle(
        Offset(rightEyeCenter.dx + offsetX, rightEyeCenter.dy + offsetY),
        pupilR,
        pupilPaint,
      );

      // Eye shine
      final shinePaint = Paint()..color = Colors.white.withOpacity(0.7);
      canvas.drawCircle(
        Offset(leftEyeCenter.dx + offsetX - s * 0.015, leftEyeCenter.dy + offsetY - s * 0.015),
        s * 0.018,
        shinePaint,
      );
      canvas.drawCircle(
        Offset(rightEyeCenter.dx + offsetX - s * 0.015, rightEyeCenter.dy + offsetY - s * 0.015),
        s * 0.018,
        shinePaint,
      );
    }

    // ── Mouth (small smile line) ───────────────────
    final mouthPaint = Paint()
      ..color = Colors.white.withOpacity(0.8)
      ..strokeWidth = s * 0.03
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;

    final mouthPath = Path();
    mouthPath.moveTo(s * 0.38, s * 0.62);
    mouthPath.quadraticBezierTo(s * 0.5, s * 0.68, s * 0.62, s * 0.62);
    canvas.drawPath(mouthPath, mouthPaint);

    // ── Ears (small bumps) ─────────────────────────
    final earPaint = Paint()..color = const Color(0xFF238B7E);
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromCenter(center: Offset(s * 0.08, s * 0.46), width: s * 0.08, height: s * 0.16),
        Radius.circular(s * 0.04),
      ),
      earPaint,
    );
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromCenter(center: Offset(s * 0.92, s * 0.46), width: s * 0.08, height: s * 0.16),
        Radius.circular(s * 0.04),
      ),
      earPaint,
    );
  }

  @override
  bool shouldRepaint(covariant _BotPainter oldDelegate) {
    return blinkValue != oldDelegate.blinkValue ||
        lookX != oldDelegate.lookX ||
        lookY != oldDelegate.lookY;
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  Simple Bot Icon (static, for FAB)
// ══════════════════════════════════════════════════════════════════════════════

class BotFab extends StatefulWidget {
  final VoidCallback onTap;
  const BotFab({super.key, required this.onTap});

  @override
  State<BotFab> createState() => _BotFabState();
}

class _BotFabState extends State<BotFab> with SingleTickerProviderStateMixin {
  late AnimationController _pulseController;
  late Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 2000),
    )..repeat(reverse: true);
    _pulseAnimation = Tween<double>(begin: 1.0, end: 1.08).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _pulseController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _pulseAnimation,
      builder: (context, child) {
        return Transform.scale(
          scale: _pulseAnimation.value,
          child: GestureDetector(
            onTap: widget.onTap,
            child: Container(
              width: 56,
              height: 56,
              decoration: BoxDecoration(
                color: const Color(0xFFFFE8D6),
                shape: BoxShape.circle,
                boxShadow: [
                  BoxShadow(
                    color: const Color(0xFFFFCBA4).withOpacity(0.5),
                    blurRadius: 12,
                    offset: const Offset(0, 4),
                  ),
                ],
              ),
              child: const Center(
                child: AnimatedBot(size: 36, isActive: true),
              ),
            ),
          ),
        );
      },
    );
  }
}
