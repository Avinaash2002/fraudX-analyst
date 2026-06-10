/// FraudX Analyst - App Provider
/// ================================
/// Manages global app state, model selection, simulation results,
/// tab navigation, and chat context

import 'package:flutter/foundation.dart';
import '../models/models.dart';
import '../services/api_service.dart';

class AppProvider with ChangeNotifier {
  // ── State ──────────────────────────────────────────────────────────────────
  List<ModelMetrics> _models = [];
  bool _isLoadingModels = false;
  String? _modelsError;

  String _selectedModel = 'LightGBM';
  PredictResponse? _lastPrediction;
  bool _isSimulating = false;
  String? _simulationError;

  // ── Tab Navigation ────────────────────────────────────────────────────────
  int _currentTabIndex = 0;
  int get currentTabIndex => _currentTabIndex;

  void switchTab(int index) {
    _currentTabIndex = index;
    notifyListeners();
  }

  // ── Chat Context (for "Ask Chatbot About This") ───────────────────────────
  String? _chatSimulationId;
  String? _pendingChatQuestion;

  String? get chatSimulationId => _chatSimulationId;
  String? get pendingChatQuestion => _pendingChatQuestion;

  void askChatbotAboutSimulation(String simulationId) {
    _chatSimulationId = simulationId;
    _pendingChatQuestion =
        'Explain my last simulation result (ID: $simulationId). '
        'Why was this transaction classified the way it was? '
        'What were the key features that influenced the prediction?';
    _currentTabIndex = 4; // Chat tab
    notifyListeners();
  }

  void clearPendingChatQuestion() {
    _pendingChatQuestion = null;
    notifyListeners();
  }

  void clearChatContext() {
    _chatSimulationId = null;
    _pendingChatQuestion = null;
  }

  // ── Getters ────────────────────────────────────────────────────────────────
  List<ModelMetrics> get models => _models;
  bool get isLoadingModels => _isLoadingModels;
  String? get modelsError => _modelsError;

  String get selectedModel => _selectedModel;
  PredictResponse? get lastPrediction => _lastPrediction;
  bool get isSimulating => _isSimulating;
  String? get simulationError => _simulationError;

  ModelMetrics? get selectedModelMetrics {
    try {
      return _models.firstWhere((m) => m.modelName == _selectedModel);
    } catch (e) {
      return null;
    }
  }

  /// Best model by F1 score
  ModelMetrics? get bestModel {
    if (_models.isEmpty) return null;
    return _models.reduce((a, b) => a.f1Score > b.f1Score ? a : b);
  }

  // ── Load Models ────────────────────────────────────────────────────────────
  Future<void> loadModels() async {
    _isLoadingModels = true;
    _modelsError = null;
    notifyListeners();

    try {
      _models = await ApiService.getModels();
      _modelsError = null;
    } catch (e) {
      _modelsError = e.toString();
      _models = [];
    } finally {
      _isLoadingModels = false;
      notifyListeners();
    }
  }

  // ── Set Selected Model ─────────────────────────────────────────────────────
  void setSelectedModel(String modelName) {
    _selectedModel = modelName;
    notifyListeners();
  }

  // ── Run Simulation ─────────────────────────────────────────────────────────
  Future<void> runSimulation(PredictRequest request) async {
    _isSimulating = true;
    _simulationError = null;
    notifyListeners();

    try {
      _lastPrediction = await ApiService.predict(request);
      _simulationError = null;
    } catch (e) {
      _simulationError = e.toString();
      _lastPrediction = null;
    } finally {
      _isSimulating = false;
      notifyListeners();
    }
  }

  // ── Clear Last Prediction ──────────────────────────────────────────────────
  void clearLastPrediction() {
    _lastPrediction = null;
    _simulationError = null;
    notifyListeners();
  }
}
