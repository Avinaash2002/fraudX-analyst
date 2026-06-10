/// FraudX Analyst - User Guide Screen
/// ======================================
/// In-app guide explaining how to use each feature

import 'package:flutter/material.dart';

class UserGuideScreen extends StatelessWidget {
  const UserGuideScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF5F7FA),
      body: SafeArea(
        child: Column(
          children: [
            // ── Header ──────────────────────────────────────────
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 16, 20, 8),
              child: Row(
                children: [
                  GestureDetector(
                    onTap: () => Navigator.of(context).pop(),
                    child: Container(
                      width: 36, height: 36,
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(10),
                        boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.05), blurRadius: 6)],
                      ),
                      child: const Icon(Icons.arrow_back_ios_new, size: 16, color: Color(0xFF1A1A2E)),
                    ),
                  ),
                  const SizedBox(width: 16),
                  const Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('User Guide', style: TextStyle(fontSize: 22, fontWeight: FontWeight.w800, color: Color(0xFF1A1A2E))),
                      Text('How to use FraudX Analyst', style: TextStyle(fontSize: 13, color: Color(0xFF6B7280))),
                    ],
                  ),
                ],
              ),
            ),

            // ── Content ─────────────────────────────────────────
            Expanded(
              child: ListView(
                padding: const EdgeInsets.all(20),
                children: const [
                  _GuideSection(
                    icon: Icons.grid_view,
                    title: 'Home Dashboard',
                    color: Color(0xFF4CAF50),
                    steps: [
                      'View your transaction protection summary at a glance.',
                      'Monitor total safe and fraud transactions detected.',
                      'Check the best model accuracy and AUC score.',
                      'Tap "View All" to see your full simulation history.',
                    ],
                  ),
                  SizedBox(height: 16),
                  _GuideSection(
                    icon: Icons.science,
                    title: 'Simulate Transaction',
                    color: Color(0xFF2196F3),
                    steps: [
                      'Select a ML model (XGBoost, LightGBM, Autoencoder, or Best Model).',
                      'Select from real transaction Dataset "Load Fraud" or "Load Normal" or "Random"',
                      'Or Enter and choose transaction details: amount, time, card number.',
                      'Tap "Analyze Transaction" to get the fraud prediction.',
                      'View the result: verdict, risk score, SHAP feature importance, and AI explanation.',
                      'Tap "Ask Chatbot About This" to get a detailed explanation from the AI assistant.',
                    ],
                  ),
                  SizedBox(height: 16),
                  _GuideSection(
                    icon: Icons.school,
                    title: 'Train / Evaluate Models',
                    color: Color(0xFFFF9800),
                    steps: [
                      'Select a built-in dataset or upload your own CSV file.',
                      'All three models (XGBoost, LightGBM, Autoencoder) will be trained.',
                      'View evaluation results: accuracy, precision, recall, F1 score, and AUC.',
                      'The best performing model is highlighted automatically.',
                      'Download a PDF report for documentation and sharing.',
                    ],
                  ),
                  SizedBox(height: 16),
                  _GuideSection(
                    icon: Icons.chat_bubble,
                    title: 'AI Chatbot (RAG)',
                    color: Color(0xFF9C27B0),
                    steps: [
                      'Ask any question about credit card fraud detection.',
                      'The chatbot uses Retrieval-Augmented Generation (RAG) to provide grounded answers.',
                      'Tap suggested questions to get started quickly.',
                      'After a simulation, ask the chatbot to explain why a transaction was flagged.',
                      'The chatbot can reference your latest simulation for context-aware answers.',
                    ],
                  ),
                  SizedBox(height: 16),
                  _GuideSection(
                    icon: Icons.bar_chart,
                    title: 'Model Comparison',
                    color: Color(0xFF2A9D8F),
                    steps: [
                      'View all available models from the MLFlow Registry.',
                      'Switch between "Models" tab (individual details) and "Comparison" tab (side-by-side).',
                      'Tap ℹ️ icons to learn what each metric (F1, AUC, etc.) means.',
                      'The best performing model is highlighted with a badge.',
                      'Download the comparison as a PDF report.',
                    ],
                  ),
                  SizedBox(height: 16),
                  _GuideSection(
                    icon: Icons.history,
                    title: 'Simulation History',
                    color: Color(0xFFEF4444),
                    steps: [
                      'Access from "View All" on the Home screen.',
                      'See all past transaction simulations with timestamps.',
                      'Tap a record to view the full prediction detail.',
                      'Swipe left on a record to delete it.',
                      'Use the trash icon to clear all history at once.',
                    ],
                  ),
                  SizedBox(height: 32),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _GuideSection extends StatelessWidget {
  final IconData icon;
  final String title;
  final Color color;
  final List<String> steps;

  const _GuideSection({
    required this.icon,
    required this.title,
    required this.color,
    required this.steps,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(color: Colors.black.withOpacity(0.04), blurRadius: 8, offset: const Offset(0, 2)),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 36, height: 36,
                decoration: BoxDecoration(
                  color: color.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Icon(icon, size: 20, color: color),
              ),
              const SizedBox(width: 12),
              Text(title, style: TextStyle(fontSize: 17, fontWeight: FontWeight.w700, color: const Color(0xFF1A1A2E))),
            ],
          ),
          const SizedBox(height: 14),
          ...steps.asMap().entries.map((entry) => Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      width: 22, height: 22,
                      margin: const EdgeInsets.only(right: 10, top: 1),
                      decoration: BoxDecoration(
                        color: color.withOpacity(0.1),
                        shape: BoxShape.circle,
                      ),
                      child: Center(
                        child: Text(
                          '${entry.key + 1}',
                          style: TextStyle(fontSize: 11, fontWeight: FontWeight.w700, color: color),
                        ),
                      ),
                    ),
                    Expanded(
                      child: Text(
                        entry.value,
                        style: const TextStyle(fontSize: 14, height: 1.4, color: Color(0xFF374151)),
                      ),
                    ),
                  ],
                ),
              )),
        ],
      ),
    );
  }
}
