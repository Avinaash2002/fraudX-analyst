/// FraudX Analyst - Tutorial Overlay (Improved)
/// ================================================
/// Full-screen overlay with step bubbles, directional arrows,
/// floating Continue button for action steps, and proper flow control.

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../services/tutorial_service.dart';
import '../providers/app_provider.dart';

class TutorialOverlay extends StatelessWidget {
  const TutorialOverlay({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<TutorialService>(
      builder: (context, tutorial, _) {
        if (!tutorial.isActive || tutorial.current == null) return const SizedBox.shrink();

        final step = tutorial.current!;
        final provider = context.read<AppProvider>();

        // Navigate to correct tab if needed
         if (provider.currentTabIndex != step.tabIndex) {
          WidgetsBinding.instance.addPostFrameCallback((_) {
            provider.switchTab(step.tabIndex);
            // Reload models when switching to Models tab
            if (step.tabIndex == 3) {
              provider.loadModels();
            }
          });
        }

        // ── When waiting for user action, show floating Continue button only ──
        if (tutorial.waitingForAction) {
          return const _FloatingContinueButton();
        }

        // ── Full overlay with bubble ──────────────────────────────────────
        return Stack(
          children: [
            // Dark overlay blocking interaction
            Positioned.fill(
              child: GestureDetector(
                onTap: () {},
                child: Container(color: Colors.black.withOpacity(0.6)),
              ),
            ),

            // Arrow indicator pointing to relevant section
            if (_getArrowPosition(context, tutorial.currentStep) != null)
              Positioned(
                top: _getArrowPosition(context, tutorial.currentStep)!['top'],
                bottom: _getArrowPosition(context, tutorial.currentStep)!['bottom'],
                left: _getArrowPosition(context, tutorial.currentStep)!['left'],
                right: _getArrowPosition(context, tutorial.currentStep)!['right'],
                child: _buildArrow(context, _getArrowDirection(tutorial.currentStep)),
              ),

            // Tutorial bubble
            Positioned.fill(
              child: SafeArea(
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 24),
                  child: Column(
                    mainAxisAlignment: _getMainAxisAlignment(step.bubbleAlignment),
                    children: [
                      if (step.bubbleAlignment == Alignment.bottomCenter ||
                          step.bubbleAlignment == Alignment.center)
                        const Spacer(),

                      // Step counter
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                        decoration: BoxDecoration(
                          color: const Color(0xFF2A9D8F).withOpacity(0.8),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Text(
                          'Step ${tutorial.currentStep + 1} of ${tutorial.steps.length}',
                          style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: Colors.white),
                        ),
                      ),
                      const SizedBox(height: 12),

                      // Bubble card
                      Container(
                        width: double.infinity,
                        padding: const EdgeInsets.all(24),
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(20),
                          boxShadow: [
                            BoxShadow(color: Colors.black.withOpacity(0.2), blurRadius: 20, offset: const Offset(0, 8)),
                          ],
                        ),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            // Title with icon
                            Row(children: [
                              Container(
                                padding: const EdgeInsets.all(8),
                                decoration: BoxDecoration(
                                  color: const Color(0xFFE0F2F1),
                                  borderRadius: BorderRadius.circular(10),
                                ),
                                child: Icon(
                                  _getStepIcon(tutorial.currentStep),
                                  size: 20,
                                  color: const Color(0xFF2A9D8F),
                                ),
                              ),
                              const SizedBox(width: 12),
                              Expanded(
                                child: Text(
                                  step.title,
                                  style: const TextStyle(
                                    fontSize: 18,
                                    fontWeight: FontWeight.w800,
                                    color: Color(0xFF1A1A2E),
                                  ),
                                ),
                              ),
                            ]),
                            const SizedBox(height: 14),

                            // Description
                            Text(
                              step.description,
                              style: const TextStyle(
                                fontSize: 14,
                                height: 1.6,
                                color: Color(0xFF4B5563),
                              ),
                            ),
                            const SizedBox(height: 20),

                            // Action hint box
                            if (step.action != null)
                              Container(
                                padding: const EdgeInsets.all(12),
                                decoration: BoxDecoration(
                                  color: const Color(0xFFFFF8E1),
                                  borderRadius: BorderRadius.circular(10),
                                  border: Border.all(color: const Color(0xFFF59E0B).withOpacity(0.3)),
                                ),
                                child: Row(children: [
                                  const Icon(Icons.touch_app, size: 18, color: Color(0xFFF59E0B)),
                                  const SizedBox(width: 8),
                                  Expanded(child: Text(
                                    _getActionHint(step.action!),
                                    style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w500, color: Color(0xFF92400E)),
                                  )),
                                ]),
                              ),

                            if (step.action != null)
                              const SizedBox(height: 12),

                            // "Let me try" button for action steps
                            if (step.action != null)
                              Padding(
                                padding: const EdgeInsets.only(bottom: 12),
                                child: SizedBox(
                                  width: double.infinity,
                                  child: GestureDetector(
                                    onTap: () {
                                      tutorial.setWaiting(true);
                                    },
                                    child: Container(
                                      padding: const EdgeInsets.symmetric(vertical: 12),
                                      decoration: BoxDecoration(
                                        color: const Color(0xFF3B82F6),
                                        borderRadius: BorderRadius.circular(12),
                                      ),
                                      child: const Center(
                                        child: Row(mainAxisSize: MainAxisSize.min, children: [
                                          Icon(Icons.touch_app, size: 18, color: Colors.white),
                                          SizedBox(width: 8),
                                          Text('Let me try', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: Colors.white)),
                                        ]),
                                      ),
                                    ),
                                  ),
                                ),
                              ),

                            // Buttons row
                            Row(
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                // Back button
                                if (tutorial.currentStep > 0)
                                  GestureDetector(
                                    onTap: () => tutorial.back(),
                                    child: Container(
                                      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                                      decoration: BoxDecoration(
                                        color: const Color(0xFFF3F4F6),
                                        borderRadius: BorderRadius.circular(12),
                                      ),
                                      child: const Row(mainAxisSize: MainAxisSize.min, children: [
                                        Icon(Icons.arrow_back, size: 16, color: Color(0xFF6B7280)),
                                        SizedBox(width: 4),
                                        Text('Back', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: Color(0xFF6B7280))),
                                      ]),
                                    ),
                                  )
                                else
                                  const SizedBox(),

                                // Next/Finish/Let me try
                                if (step.action == null)
                                  GestureDetector(
                                    onTap: () => tutorial.next(),
                                    child: Container(
                                      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
                                      decoration: BoxDecoration(
                                        color: const Color(0xFF2A9D8F),
                                        borderRadius: BorderRadius.circular(12),
                                      ),
                                      child: Row(mainAxisSize: MainAxisSize.min, children: [
                                        Text(
                                          tutorial.currentStep == tutorial.steps.length - 1 ? 'Finish' : 'Next',
                                          style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w700, color: Colors.white),
                                        ),
                                        if (tutorial.currentStep < tutorial.steps.length - 1) ...[
                                          const SizedBox(width: 4),
                                          const Icon(Icons.arrow_forward, size: 16, color: Colors.white),
                                        ],
                                      ]),
                                    ),
                                  ),
                              ],
                            ),
                          ],
                        ),
                      ),

                      if (step.bubbleAlignment == Alignment.topCenter)
                        const Spacer(),
                    ],
                  ),
                ),
              ),
            ),
          ],
        );
      },
    );
  }

  // ── Arrow indicator widget ────────────────────────────────────
  Widget _buildArrow(BuildContext context, String direction) {
    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0.3, end: 1.0),
      duration: const Duration(milliseconds: 600),
      curve: Curves.easeInOut,
      builder: (context, value, child) {
        return Opacity(
          opacity: value,
          child: Container(
            width: 22,
            height: 22,
            decoration: BoxDecoration(
              color: const Color(0xFFFFE500),
              shape: BoxShape.circle,
              boxShadow: [
                BoxShadow(
                  color: const Color(0xFFFFE500).withOpacity(0.8),
                  blurRadius: 16,
                  spreadRadius: 4,
                ),
                BoxShadow(
                  color: const Color(0xFFFFE500).withOpacity(0.4),
                  blurRadius: 30,
                  spreadRadius: 8,
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  // ── Arrow positions for each step ─────────────────────────────
  Map<String, double?>? _getArrowPosition(BuildContext context, int step) {
    final screenHeight = MediaQuery.of(context).size.height;
    final screenWidth = MediaQuery.of(context).size.width;

    switch (step) {
      case 0: // Hero card — near "Protection Active" top right
        return {'top': screenHeight * 0.17, 'bottom': null, 'left': null, 'right': screenWidth * 0.12};
      case 1: // Stats grid — between Safe Today and Fraud Blocked
        return {'top': screenHeight * 0.48, 'bottom': null, 'left': screenWidth * 0.47, 'right': null};
      case 2: // Recent Transactions — near the "Recent Transactions" text
        return {'top': screenHeight * 0.86, 'bottom': null, 'left': screenWidth * 0.52, 'right': null};
      case 3: return null; // FAB flashes instead
      case 4: // Model selector — top of simulate page
        return {'top': screenHeight * 0.185, 'bottom': null, 'left': screenWidth * 0.42, 'right': null};
      case 5: // Load dataset buttons — below model selector
        return {'top': screenHeight * 0.35, 'bottom': null, 'left': screenWidth * 0.43, 'right': null};
      case 6: // Transaction details — middle of page
        return {'top': screenHeight * 0.53, 'bottom': null, 'left': screenWidth * 0.49, 'right': null};
      default: return null;
    }
  }

  String _getArrowDirection(int step) {
    switch (step) {
      case 0: return 'up';     // Points up to hero card
      case 1: return 'up';     // Points up to stats
      case 2: return 'down';   // Points down to recent
      case 3: return 'right';  // Points right to FAB
      case 4: return 'up';     // Points up to model selector
      case 5: return 'up';     // Points up to load dataset
      case 6: return 'up';     // Points up to transaction details
      default: return 'down';
    }
  }

  MainAxisAlignment _getMainAxisAlignment(Alignment alignment) {
    if (alignment == Alignment.topCenter) return MainAxisAlignment.start;
    if (alignment == Alignment.bottomCenter) return MainAxisAlignment.end;
    return MainAxisAlignment.center;
  }

  IconData _getStepIcon(int step) {
    const icons = [
      Icons.shield_outlined,      // 0: Protection overview
      Icons.dashboard,             // 1: Stats cards
      Icons.history,               // 2: Recent transactions
      Icons.smart_toy,             // 3: Chatbot icon
      Icons.psychology,            // 4: ML Model selector
      Icons.dataset,               // 5: Load from dataset
      Icons.edit_note,             // 6: Transaction details
      Icons.play_arrow,            // 7: Try it out
      Icons.analytics,             // 8: Result
      Icons.chat,                  // 9: Chatbot page
      Icons.question_answer,       // 10: Follow-up
      Icons.compare_arrows,        // 11: Model comparison
      Icons.info_outline,          // 12: Model details
      Icons.model_training,        // 13: Train
      Icons.table_chart,           // 14: Dataset format
      Icons.celebration,           // 15: Finish
    ];
    return step < icons.length ? icons[step] : Icons.circle;
  }

  String _getActionHint(String action) {
    switch (action) {
      case 'auto_advance':
        return 'Get ready — the simulation page will be yours in a moment!';
      case 'prompt_simulate_and_ask':
        return 'Load a transaction → Analyze → then tap "Ask Chatbot About This" on the result';
      case 'prompt_wait_chatbot':
        return 'Read the chatbot\'s response, then press Continue when you\'re ready';
      case 'prompt_simpler_terms':
        return 'Type "Explain to me in simpler terms" and send it. Wait for the response.';
      case 'prompt_dataset_format':
        return 'Tap the glowing button to view the required format, then swipe down to close it';
      case 'prompt_model_info':
        return 'Tap any glowing "i" icon to view the metric explanation, then close it';
      default:
        return 'Follow the instruction above, then press Continue when ready.';
    }
  }
}

class _FloatingContinueButton extends StatelessWidget {
  const _FloatingContinueButton();

  @override
  Widget build(BuildContext context) {
    final tutorial = context.watch<TutorialService>();

    // Only show when action is completed
    if (!tutorial.actionCompleted) return const SizedBox.shrink();

    return Positioned(
      bottom: 100,
      left: 0,
      right: 0,
      child: Center(
        child: GestureDetector(
          onTap: () {
            tutorial.setWaiting(false);
            tutorial.next();
          },
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 14),
            decoration: BoxDecoration(
              color: const Color(0xFF2A9D8F),
              borderRadius: BorderRadius.circular(30),
              boxShadow: [
                BoxShadow(color: const Color(0xFF2A9D8F).withOpacity(0.4), blurRadius: 16, offset: const Offset(0, 6)),
              ],
            ),
            child: const Row(mainAxisSize: MainAxisSize.min, children: [
              Icon(Icons.check_circle, size: 20, color: Colors.white),
              SizedBox(width: 8),
              Text('Continue', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: Colors.white)),
              SizedBox(width: 4),
              Icon(Icons.arrow_forward, size: 18, color: Colors.white),
            ]),
          ),
        ),
      ),
    );
  }
}