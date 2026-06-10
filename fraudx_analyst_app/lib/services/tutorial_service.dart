/// FraudX Analyst - Interactive Tutorial Service (v2)
/// ===================================================
/// Fixed step flow: Continue button for action steps,
/// proper waiting for user interaction.

import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

class TutorialStep {
  final String title;
  final String description;
  final int tabIndex;
  final String? action;
  final bool waitForResult;
  final Alignment bubbleAlignment;

  const TutorialStep({
    required this.title,
    required this.description,
    required this.tabIndex,
    this.action,
    this.waitForResult = false,
    this.bubbleAlignment = Alignment.center,
  });
}

class TutorialService extends ChangeNotifier {
  bool _isActive = false;
  int _currentStep = 0;
  bool _waitingForAction = false;
  bool _actionCompleted = false;

  bool get isActive => _isActive;
  int get currentStep => _currentStep;
  bool get waitingForAction => _waitingForAction;
  bool get actionCompleted => _actionCompleted;
  TutorialStep? get current => _isActive && _currentStep < steps.length ? steps[_currentStep] : null;

  static const String _prefKey = 'has_completed_tutorial';

  final List<TutorialStep> steps = [
    // ── HOME PAGE (Steps 0-3) ────────────────────────────────
    const TutorialStep(
      title: 'Protection Overview',
      description: 'Your cumulative sum of simulated transactions analysed is displayed here. This value updates as you run more simulations.',
      tabIndex: 0,
      bubbleAlignment: Alignment.bottomCenter,
    ),
    const TutorialStep(
      title: 'Dashboard Statistics',
      description: 'These cards show your total safe and fraud transactions, the best model\'s accuracy, and the model AUC score. All values update in real time from the backend.',
      tabIndex: 0,
      bubbleAlignment: Alignment.center,
    ),
    const TutorialStep(
      title: 'Recent Transactions',
      description: 'Your simulation history appears here. Since this is your first time, it\'s empty — make your first simulation to see results!',
      tabIndex: 0,
      bubbleAlignment: Alignment.topCenter,
    ),
    const TutorialStep(
      title: 'Chatbot Access',
      description: 'You can access the AI-powered fraud detection chatbot anytime by tapping this floating icon. It uses RAG (Retrieval-Augmented Generation) to answer your questions.',
      tabIndex: 0,
      bubbleAlignment: Alignment.topCenter,
    ),

    // ── SIMULATE PAGE (Steps 4-6) ────────────────────────────
    const TutorialStep(
      title: 'Select ML Model',
      description: 'This application has 3 different machine learning models you can test:\n\n1. XGBoost (Supervised)\n2. LightGBM (Supervised)\n3. Autoencoder (Unsupervised)\n\nBy default, the best-performing model is selected automatically.',
      tabIndex: 1,
      bubbleAlignment: Alignment.bottomCenter,
    ),
    const TutorialStep(
      title: 'Load from Dataset',
      description: 'Load real transactions that the model has never seen during training. Since we cannot simulate actual credit card purchases for security reasons, these buttons load real historical patterns (card usage, cardholder profile, geographic consistency, and more) from a held-out test set to test the model in a real-world scenario.',
      tabIndex: 1,
      bubbleAlignment: Alignment.center,
    ),
    const TutorialStep(
      title: 'Transaction Details',
      description: 'Enter a random amount, time, and card number here. The hidden features like card usage patterns and cardholder profile are loaded automatically when you use "Load from Dataset" or generated randomly.',
      tabIndex: 1,
      bubbleAlignment: Alignment.topCenter,
    ),

    // Step 7: Try It Out intro then user clicks Next
    const TutorialStep(
      title: 'Try It Out!',
      description: 'Let\'s run your first fraud detection simulation. You\'ll load a real transaction and see the model predict whether it\'s fraud or normal.',
      tabIndex: 1,
      bubbleAlignment: Alignment.center,
    ),

    // Step 8: User does simulation + asks chatbot
    const TutorialStep(
      title: 'Simulation Result',
      description: 'Load a transaction (Fraud/Normal/Random), tap "Analyze Transaction", then scroll down on the result sheet and tap "Ask Chatbot About This".',
      tabIndex: 1,
      action: 'prompt_simulate_and_ask',
      waitForResult: true,
      bubbleAlignment: Alignment.center,
    ),

    // ── CHAT PAGE (Steps 9-10) ───────────────────────────────
    const TutorialStep(
      title: 'FraudX AI Chatbot',
      description: 'This chatbot is powered by RAG (Retrieval-Augmented Generation) using Google Gemini and a Pinecone knowledge base. You can ask it anything about fraud detection, model explanations, or your simulation results.\n\nWait for the chatbot to respond to your simulation question, then press Next.',
      tabIndex: 4,
      bubbleAlignment: Alignment.center,
    ),

    const TutorialStep(
      title: 'Ask a Follow-Up',
      description: 'Try asking: "Explain to me in simpler terms" — this shows how the chatbot maintains conversation context and provides clearer explanations.',
      tabIndex: 4,
      action: 'prompt_simpler_terms',
      waitForResult: true,
      bubbleAlignment: Alignment.center,
    ),

    // ── MODELS PAGE (Steps 11-12) ────────────────────────────
    const TutorialStep(
      title: 'Model Comparison',
      description: 'Here you can see a side-by-side comparison of all available AI models in the system. The bar charts show how each model performs across different evaluation metrics.',
      tabIndex: 3,
      bubbleAlignment: Alignment.center,
    ),
    const TutorialStep(
      title: 'Explore Model Details',
      description: 'Tap any glowing "i" icon to learn what each metric means.',
      tabIndex: 3,
      action: 'prompt_model_info',
      waitForResult: true,
      bubbleAlignment: Alignment.center,
    ),

    // ── TRAIN PAGE (Steps 13-14) ─────────────────────────────
    const TutorialStep(
      title: 'Train Models',
      description: 'You can train the models using the built-in dataset containing 284,807 real transactions, or upload your own custom CSV dataset.',
      tabIndex: 2,
      bubbleAlignment: Alignment.center,
    ),
    const TutorialStep(
      title: 'Dataset Format',
      description: 'Tap the glowing button to see what columns are required for a custom dataset.',
      tabIndex: 2,
      action: 'prompt_dataset_format',
      waitForResult: true,
      bubbleAlignment: Alignment.center,
    ),

    // ── FINISH (Step 15) ─────────────────────────────────────
    const TutorialStep(
      title: 'You\'re All Set! 🎉',
      description: 'Thank you for completing the guide! I hope you enjoy discovering more about credit card fraud detection through this application.\n\nEnjoy exploring FraudX Analyst!',
      tabIndex: 0,
      bubbleAlignment: Alignment.center,
    ),
  ];

  Future<bool> shouldShowTutorial() async {
    final prefs = await SharedPreferences.getInstance();
    return !(prefs.getBool(_prefKey) ?? false);
  }

  void start() {
    _isActive = true;
    _currentStep = 0;
    _waitingForAction = false;
    notifyListeners();
  }

  void next() {
     if (_currentStep < steps.length - 1) {
      _currentStep++;
      _waitingForAction = false;
      _actionCompleted = false;
      notifyListeners();
    } else {
      finish();
    }
  }

  void back() {
    if (_currentStep > 0) {
      _currentStep--;
      _waitingForAction = false;
      _actionCompleted = false;
      notifyListeners();
    }
  }

  void setWaiting(bool waiting) {
    _waitingForAction = waiting;
    _actionCompleted = false;
    notifyListeners();
  }

  void completeAction() {
    _actionCompleted = true;
    notifyListeners();
  }

  Future<void> finish() async {
    _isActive = false;
    _currentStep = 0;
    _waitingForAction = false;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_prefKey, true);
    notifyListeners();
  }

  Future<void> reset() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_prefKey);
    _isActive = false;
    _currentStep = 0;
    notifyListeners();
  }
}
