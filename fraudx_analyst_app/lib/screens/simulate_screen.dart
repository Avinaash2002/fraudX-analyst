/// FraudX Analyst - Simulate Screen
/// ===================================

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:math';
import 'package:provider/provider.dart';
import '../providers/app_provider.dart';
import '../models/models.dart';
import '../config/api_config.dart';
import '../services/api_service.dart';
import '../services/tutorial_service.dart';

class SimulateScreen extends StatefulWidget {
  const SimulateScreen({super.key});
  @override
  State<SimulateScreen> createState() => _SimulateScreenState();
}

class _SimulateScreenState extends State<SimulateScreen> {
  final _amountController = TextEditingController();
  final _timeController = TextEditingController();
  final _cardController = TextEditingController();

  // Inline error messages (null = no error)
  String? _amountError;
  String? _cardError;

  String _selectedModel = 'Best Model';

  final Map<String, String> _modelDescriptions = {
    'Best Model': 'Auto-select highest F1 score',
    'XGBoost': 'Gradient boosting ensemble',
    'LightGBM': 'Light gradient boosting',
    'Autoencoder': 'Deep learning anomaly',
  };

  // ── Loaded sample features from real dataset ────────────────────────────────
  Map<String, double>? _loadedFeatures;
  String? _loadedType;
  bool _isLoadingSample = false;
  int _loadCounter = 0;
  TimeOfDay _selectedTime = const TimeOfDay(hour: 14, minute: 30);

  @override
  void initState() {
    super.initState();
    // Real-time validation listeners
    _amountController.addListener(_validateAmount);
    _cardController.addListener(_validateCard);
  }

  void _validateAmount() {
    final text = _amountController.text;
    if (text.isEmpty) {
      setState(() => _amountError = null); // Don't show error for empty (not started)
      return;
    }
    final value = double.tryParse(text);
    if (value == null || value <= 0) {
      setState(() => _amountError = 'Enter a valid amount');
    } else {
      setState(() => _amountError = null);
    }
  }

  void _validateCard() {
    final digits = _cardController.text.replaceAll(' ', '');
    if (digits.isEmpty) {
      setState(() => _cardError = null);
      return;
    }
    if (digits.length < 16) {
      setState(() => _cardError = 'Invalid card number');
    } else {
      setState(() => _cardError = null);
    }
  }

  // ── Time Picker ──────────────────────────────────────────────
  Future<void> _pickTime() async {
    final picked = await showTimePicker(
      context: context,
      initialTime: _selectedTime,
      builder: (context, child) {
        return Theme(
          data: Theme.of(context).copyWith(
            colorScheme: const ColorScheme.light(primary: Color(0xFF2A9D8F)),
          ),
          child: child!,
        );
      },
    );
    if (picked != null) {
      setState(() {
        _selectedTime = picked;
        _timeController.text = '${picked.hour.toString().padLeft(2, '0')}:${picked.minute.toString().padLeft(2, '0')}';
      });
    }
  }

  Future<void> _loadSampleTransaction(String type) async {
    _loadCounter++;
    final thisLoad = _loadCounter;
    setState(() {
      _isLoadingSample = true;
      _loadedFeatures = null; // Clear previous features immediately
      _loadedType = null;
    });
    try {
      final data = await ApiService.getSampleTransaction(type);
      if (mounted && thisLoad == _loadCounter) { // Only use if this is still the latest load
        final features = <String, double>{};
        for (int i = 1; i <= 28; i++) {
          features['v$i'] = (data['v$i'] as num).toDouble();
        }
        setState(() {
          _loadedFeatures = features;
          _loadedType = data['type'] as String;
          _amountController.text = (data['amount'] as num).toStringAsFixed(2);
          final totalSec = (data['time'] as num).toDouble();
          final hours = (totalSec / 3600).floor() % 24;
          final mins = ((totalSec % 3600) / 60).floor();
          _selectedTime = TimeOfDay(hour: hours, minute: mins);
          _timeController.text = '${hours.toString().padLeft(2, '0')}:${mins.toString().padLeft(2, '0')}';
          if (_cardController.text.isEmpty) _cardController.text = '4532 8721 0945 6138';
          _isLoadingSample = false;
        });
      }
    } catch (e) {
      if (mounted && thisLoad == _loadCounter) {
        setState(() => _isLoadingSample = false);
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text('Error: $e'), backgroundColor: const Color(0xFFEF4444),
        ));
      }
    }
  }

  @override
  void dispose() {
    _amountController.removeListener(_validateAmount);
    _cardController.removeListener(_validateCard);
    _amountController.dispose();
    _timeController.dispose();
    _cardController.dispose();
    super.dispose();
  }

  String _resolveModelName(AppProvider provider) {
    if (_selectedModel == 'Best Model') {
      return provider.bestModel?.modelName ?? 'LightGBM';
    }
    return _selectedModel;
  }

  Future<void> _analyzeTransaction() async {
    // Validate amount
    if (_amountController.text.trim().isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Please enter a transaction amount'), backgroundColor: Color(0xFFFF9800)));
      return;
    }
    // Validate time
    if (_timeController.text.trim().isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Please select a transaction time'), backgroundColor: Color(0xFFFF9800)));
      return;
    }
    // Validate card number (must be 16 digits)
    final cardDigits = _cardController.text.replaceAll(' ', '');
    if (cardDigits.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Please enter a card number'), backgroundColor: Color(0xFFFF9800)));
      return;
    }
    if (cardDigits.length != 16) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Card number must be 16 digits (currently ${cardDigits.length})'), backgroundColor: const Color(0xFFFF9800)));
      return;
    }
    final provider = context.read<AppProvider>();
    final timeParts = _timeController.text.split(':');
    double timeSeconds = 0;
    if (timeParts.length == 2) {
      timeSeconds = (double.tryParse(timeParts[0]) ?? 0) * 3600 + (double.tryParse(timeParts[1]) ?? 0) * 60;
    }

    // Use loaded real features if available, otherwise random
    final features = <String, double>{};
    if (_loadedFeatures != null) {
      features.addAll(_loadedFeatures!);
      _loadedFeatures = null; // Clear after use
    } else {
      final random = Random();
      for (int i = 1; i <= 28; i++) {
        features['v$i'] = (random.nextDouble() * 30) - 15;
      }
    }
    final request = PredictRequest(
      modelName: _resolveModelName(provider),
      amount: double.tryParse(_amountController.text) ?? 0,
      time: timeSeconds,
      features: features,
      deviceId: ApiConfig.deviceId,
      cardNumber: _cardController.text,
      location: "",
    );
    await provider.runSimulation(request);
    if (mounted) {
      if (provider.lastPrediction != null) {
        _showResultSheet(context, provider.lastPrediction!);
        // Tell tutorial the simulation is done
        try {
          final tutorial = context.read<TutorialService>();
          if (tutorial.isActive && tutorial.waitingForAction) {
            tutorial.completeAction();
          }
        } catch (_) {}
      }
    }
  }

  void _showResultSheet(BuildContext context, PredictResponse result) {
    final provider = context.read<AppProvider>();
    bool isTutorialActive = false;
    try {
      isTutorialActive = context.read<TutorialService>().isActive;
    } catch (_) {}
    showModalBottomSheet(
      context: context, isScrollControlled: true, backgroundColor: Colors.transparent,
      isDismissible: !isTutorialActive,
      enableDrag: !isTutorialActive,
      useRootNavigator: false,
      builder: (ctx) => _ResultSheet(
        result: result,
        isTutorialActive: isTutorialActive,
        onAskChatbot: () {
          Navigator.pop(ctx);
          WidgetsBinding.instance.addPostFrameCallback((_) {
            provider.askChatbotAboutSimulation(result.simulationId);
            // Advance tutorial after Ask Chatbot
            try {
              final tutorial = context.read<TutorialService>();
              if (tutorial.isActive && tutorial.waitingForAction) {
                tutorial.completeAction();
                Future.delayed(const Duration(milliseconds: 500), () {
                  tutorial.setWaiting(false);
                  tutorial.next();
                });
              }
            } catch (_) {}
          });
        },
      ),
    );
  }

  void _showModelPicker() {
    showModalBottomSheet(
      context: context, backgroundColor: Colors.white,
      shape: const RoundedRectangleBorder(borderRadius: BorderRadius.vertical(top: Radius.circular(20))),
      builder: (ctx) => Padding(
        padding: const EdgeInsets.all(20),
        child: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.start, children: [
          const Text('Select ML Model', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700, color: Color(0xFF1A1A2E))),
          const SizedBox(height: 16),
          ..._modelDescriptions.entries.map((entry) => ListTile(
            onTap: () { setState(() => _selectedModel = entry.key); Navigator.pop(ctx); },
            leading: Container(width: 40, height: 40, decoration: BoxDecoration(color: const Color(0xFFE0F2F1), borderRadius: BorderRadius.circular(12)),
              child: Icon(entry.key == 'Best Model' ? Icons.auto_awesome : Icons.psychology, size: 20, color: const Color(0xFF2A9D8F))),
            title: Text(entry.key, style: TextStyle(fontWeight: FontWeight.w600, color: _selectedModel == entry.key ? const Color(0xFF2A9D8F) : const Color(0xFF1A1A2E))),
            subtitle: Text(entry.value),
            trailing: _selectedModel == entry.key ? const Icon(Icons.check_circle, color: Color(0xFF2A9D8F)) : null,
          )),
          const SizedBox(height: 12),
        ]),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final isSimulating = context.watch<AppProvider>().isSimulating;
    return Scaffold(
      backgroundColor: const Color(0xFFF5F7FA),
      body: SafeArea(
        child: SingleChildScrollView(
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            // ── Header ─────────────────────────────────────────────
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 16, 20, 24),
              child: Row(children: [
                Container(width: 36, height: 36, decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(10), boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.05), blurRadius: 6)]),
                  child: const Icon(Icons.arrow_back_ios_new, size: 16, color: Color(0xFF1A1A2E))),
                const SizedBox(width: 16),
                const Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text('Simulate\nTransaction', style: TextStyle(fontSize: 24, fontWeight: FontWeight.w800, color: Color(0xFF1A1A2E), height: 1.15)),
                  SizedBox(height: 4),
                  Text('Test fraud detection models', style: TextStyle(fontSize: 14, color: Color(0xFF6B7280))),
                ]),
              ]),
            ),
            Padding(padding: const EdgeInsets.symmetric(horizontal: 20), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              // ── Model Selector ──────────────────────────────────
              const Text('Select ML Model', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600, color: Color(0xFF1A1A2E))),
              const SizedBox(height: 10),
              GestureDetector(
                onTap: _showModelPicker,
                child: Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(16), boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.04), blurRadius: 8, offset: const Offset(0, 2))]),
                  child: Row(children: [
                    Container(width: 40, height: 40, decoration: BoxDecoration(color: const Color(0xFFE0F2F1), borderRadius: BorderRadius.circular(12)),
                      child: Icon(_selectedModel == 'Best Model' ? Icons.auto_awesome : Icons.psychology, size: 20, color: const Color(0xFF2A9D8F))),
                    const SizedBox(width: 14),
                    Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                      Text(_selectedModel, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: Color(0xFF1A1A2E))),
                      Text(_modelDescriptions[_selectedModel] ?? '', style: const TextStyle(fontSize: 13, color: Color(0xFF6B7280))),
                    ])),
                    const Icon(Icons.keyboard_arrow_down, color: Color(0xFF6B7280)),
                  ]),
                ),
              ),
              const SizedBox(height: 28),

              // ── Load from Dataset ──────────────────────────────
              const Text('Load from Dataset', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600, color: Color(0xFF1A1A2E))),
              const SizedBox(height: 4),
              Text('Load real transactions from the Kaggle dataset', style: TextStyle(fontSize: 12, color: Colors.grey.shade500)),
              const SizedBox(height: 10),
              Row(children: [
                Expanded(child: _LoadButton(
                  label: 'Load Fraud',
                  icon: Icons.warning_amber,
                  color: const Color(0xFFEF4444),
                  isLoading: _isLoadingSample,
                  onTap: () => _loadSampleTransaction('fraud'),
                )),
                const SizedBox(width: 10),
                Expanded(child: _LoadButton(
                  label: 'Load Normal',
                  icon: Icons.check_circle_outline,
                  color: const Color(0xFF2A9D8F),
                  isLoading: _isLoadingSample,
                  onTap: () => _loadSampleTransaction('normal'),
                )),
                const SizedBox(width: 10),
                Expanded(child: _LoadButton(
                  label: 'Random',
                  icon: Icons.shuffle,
                  color: const Color(0xFF3B82F6),
                  isLoading: _isLoadingSample,
                  onTap: () => _loadSampleTransaction('random'),
                )),
              ]),
              if (_loadedType != null) ...[
                const SizedBox(height: 8),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                  decoration: BoxDecoration(
                    color: _loadedType == 'FRAUD' ? const Color(0xFFFFEBEE) : const Color(0xFFE8F5E9),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Row(children: [
                    Icon(
                      _loadedType == 'FRAUD' ? Icons.warning_amber : Icons.check_circle,
                      size: 16,
                      color: _loadedType == 'FRAUD' ? const Color(0xFFD32F2F) : const Color(0xFF2E7D32),
                    ),
                    const SizedBox(width: 8),
                    Text(
                      'Real ${_loadedType} from test set (unseen by model)',
                      style: TextStyle(fontSize: 12, fontWeight: FontWeight.w500, color: _loadedType == 'FRAUD' ? const Color(0xFFD32F2F) : const Color(0xFF2E7D32)),
                    ),
                  ]),
                ),
              ],
              const SizedBox(height: 24),

              // ── Transaction Details ──────────────────────────────
              const Text('Transaction Details', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700, color: Color(0xFF1A1A2E))),
              const SizedBox(height: 16),

              // Amount — numbers only, max 2 decimal places
              const Text('Amount (\$)', style: TextStyle(fontSize: 13, fontWeight: FontWeight.w500, color: Color(0xFF6B7280))),
              const SizedBox(height: 6),
              Container(
                decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(14),
                  border: _amountError != null ? Border.all(color: const Color(0xFFEF4444), width: 1.5) : null,
                  boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.04), blurRadius: 8, offset: const Offset(0, 2))]),
                child: TextField(controller: _amountController,
                  keyboardType: const TextInputType.numberWithOptions(decimal: true),
                  inputFormatters: [FilteringTextInputFormatter.allow(RegExp(r'^\d+\.?\d{0,2}')),],
                  style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w500, color: Color(0xFF1A1A2E)),
                  decoration: InputDecoration(prefixIcon: const Icon(Icons.attach_money, size: 20, color: Color(0xFF9CA3AF)), hintText: 'e.g. 1250.00', hintStyle: const TextStyle(fontSize: 14, color: Color(0xFFBBBBCC)), border: OutlineInputBorder(borderRadius: BorderRadius.circular(14), borderSide: BorderSide.none), filled: true, fillColor: Colors.white, contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14))),
              ),
              if (_amountError != null)
                Padding(padding: const EdgeInsets.only(top: 4, left: 4), child: Text(_amountError!, style: const TextStyle(fontSize: 12, color: Color(0xFFEF4444)))),
              const SizedBox(height: 16),

              Row(children: [
                // Time — tap to pick
                Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  const Text('Time', style: TextStyle(fontSize: 13, fontWeight: FontWeight.w500, color: Color(0xFF6B7280))),
                  const SizedBox(height: 6),
                  GestureDetector(
                    onTap: _pickTime,
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
                      decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(14), boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.04), blurRadius: 8, offset: const Offset(0, 2))]),
                      child: Row(children: [
                        const Icon(Icons.access_time, size: 20, color: Color(0xFF9CA3AF)),
                        const SizedBox(width: 12),
                        Text(_timeController.text.isEmpty ? 'Select time' : _timeController.text, style: TextStyle(fontSize: 15, fontWeight: FontWeight.w500, color: _timeController.text.isEmpty ? const Color(0xFFBBBBCC) : const Color(0xFF1A1A2E))),
                      ]),
                    ),
                  ),
                ])),
                const SizedBox(width: 12),

                // Card — 16 digits with auto-spacing
                Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  const Text('Card Number', style: TextStyle(fontSize: 13, fontWeight: FontWeight.w500, color: Color(0xFF6B7280))),
                  const SizedBox(height: 6),
                  Container(
                    decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(14),
                      border: _cardError != null ? Border.all(color: const Color(0xFFEF4444), width: 1.5) : null,
                      boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.04), blurRadius: 8, offset: const Offset(0, 2))]),
                    child: TextField(controller: _cardController,
                      keyboardType: TextInputType.number,
                      maxLength: 19, // 16 digits + 3 spaces
                      inputFormatters: [FilteringTextInputFormatter.digitsOnly, _CardNumberFormatter()],
                      style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w500, color: Color(0xFF1A1A2E)),
                      decoration: InputDecoration(prefixIcon: const Icon(Icons.credit_card, size: 20, color: Color(0xFF9CA3AF)), hintText: '0000 0000 0000 0000', hintStyle: const TextStyle(fontSize: 14, color: Color(0xFFBBBBCC)), counterText: '', border: OutlineInputBorder(borderRadius: BorderRadius.circular(14), borderSide: BorderSide.none), filled: true, fillColor: Colors.white, contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14))),
                  ),
                  if (_cardError != null)
                    Padding(padding: const EdgeInsets.only(top: 4, left: 4), child: Text(_cardError!, style: const TextStyle(fontSize: 12, color: Color(0xFFEF4444)))),
                ])),
              ]),
              const SizedBox(height: 16),

             
              // ── Analyze Button ──────────────────────────────────
              SizedBox(width: double.infinity, height: 56,
                child: ElevatedButton(
                  onPressed: isSimulating ? null : _analyzeTransaction,
                  style: ElevatedButton.styleFrom(backgroundColor: const Color(0xFF2A9D8F), foregroundColor: Colors.white, disabledBackgroundColor: const Color(0xFF2A9D8F).withOpacity(0.5), elevation: 0, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16))),
                  child: isSimulating
                      ? const Row(mainAxisAlignment: MainAxisAlignment.center, children: [SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2.5, color: Colors.white)), SizedBox(width: 12), Text('Analyzing…', style: TextStyle(fontSize: 17, fontWeight: FontWeight.w700))])
                      : const Row(mainAxisAlignment: MainAxisAlignment.center, children: [Icon(Icons.play_arrow, size: 22), SizedBox(width: 8), Text('Analyze Transaction', style: TextStyle(fontSize: 17, fontWeight: FontWeight.w700))]),
                ),
              ),
              const SizedBox(height: 24),
            ])),
          ]),
        ),
      ),
    );
  }
}

class _InputField extends StatelessWidget {
  final TextEditingController controller; final IconData prefixIcon; final TextInputType? keyboardType; final String? hintText;
  const _InputField({required this.controller, required this.prefixIcon, this.keyboardType, this.hintText});
  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(14), boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.04), blurRadius: 8, offset: const Offset(0, 2))]),
      child: TextField(controller: controller, keyboardType: keyboardType,
        style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w500, color: Color(0xFF1A1A2E)),
        decoration: InputDecoration(prefixIcon: Icon(prefixIcon, size: 20, color: const Color(0xFF9CA3AF)), hintText: hintText, hintStyle: const TextStyle(fontSize: 14, color: Color(0xFFBBBBCC)), border: OutlineInputBorder(borderRadius: BorderRadius.circular(14), borderSide: BorderSide.none), filled: true, fillColor: Colors.white, contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14))),
    );
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  Result Sheet — with Sim ID, DateTime, SHAP bars, Ask Chatbot button
// ══════════════════════════════════════════════════════════════════════════════

class _ResultSheet extends StatelessWidget {
  final PredictResponse result;
  final VoidCallback onAskChatbot;
  final bool isTutorialActive;
  const _ResultSheet({required this.result, required this.onAskChatbot, this.isTutorialActive = false});

  @override
  Widget build(BuildContext context) {
    final isFraud = result.isFraud;
    return PopScope(
      canPop: !isTutorialActive,
      child: GestureDetector(
        onVerticalDragUpdate: isTutorialActive ? (_) {} : null,
        child: DraggableScrollableSheet(
          initialChildSize: 0.85,
          minChildSize: isTutorialActive ? 0.85 : 0.5,
          maxChildSize: isTutorialActive ? 0.85 : 0.95,
        builder: (ctx, scrollController) {
        return Container(
          decoration: const BoxDecoration(color: Colors.white, borderRadius: BorderRadius.vertical(top: Radius.circular(24))),
          child: SingleChildScrollView(
            controller: scrollController,
            padding: const EdgeInsets.all(24),
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Center(child: Container(width: 40, height: 4, decoration: BoxDecoration(color: Colors.grey.shade300, borderRadius: BorderRadius.circular(2)))),
              const SizedBox(height: 20),

              // ── Verdict Badge ──────────────────────────────────
              Center(child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                decoration: BoxDecoration(color: isFraud ? const Color(0xFFFFEBEE) : const Color(0xFFE8F5E9), borderRadius: BorderRadius.circular(20)),
                child: Row(mainAxisSize: MainAxisSize.min, children: [
                  Icon(isFraud ? Icons.warning_amber : Icons.check_circle, color: isFraud ? const Color(0xFFD32F2F) : const Color(0xFF2E7D32)),
                  const SizedBox(width: 8),
                  Text(isFraud ? 'FRAUD DETECTED' : 'SAFE TRANSACTION', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: isFraud ? const Color(0xFFD32F2F) : const Color(0xFF2E7D32))),
                ]),
              )),
              const SizedBox(height: 20),

              // ── Simulation ID & DateTime ───────────────────────
              Container(
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(color: const Color(0xFFF5F7FA), borderRadius: BorderRadius.circular(12)),
                child: Row(children: [
                  Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    const Text('Simulation ID', style: TextStyle(fontSize: 11, color: Color(0xFF9CA3AF))),
                    const SizedBox(height: 2),
                    Text(result.simulationId.length > 12 ? result.simulationId.substring(0, 12) : result.simulationId, style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: Color(0xFF1A1A2E))),
                  ])),
                  Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.end, children: [
                    const Text('Date / Time', style: TextStyle(fontSize: 11, color: Color(0xFF9CA3AF))),
                    const SizedBox(height: 2),
                    Text(_formatNow(), style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: Color(0xFF1A1A2E))),
                  ])),
                ]),
              ),
              const SizedBox(height: 16),

              // ── Scores ─────────────────────────────────────────
              _ResultRow(label: 'Risk Score', value: '${(result.riskScore * 100).toStringAsFixed(2)}%'),
              _ResultRow(label: 'Confidence', value: '${(result.confidenceScore * 100).toStringAsFixed(2)}%'),
              _ResultRow(label: 'Processing Time', value: '${result.processingTime.toStringAsFixed(0)}ms'),
              const SizedBox(height: 20),

              // ── SHAP Feature Importance Bar Chart ──────────────
              const Text('Feature Importance', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: Color(0xFF1A1A2E))),
              const SizedBox(height: 12),
              ...result.topFeatures.take(7).map((f) {
                final absImpact = f.impact.abs();
                final maxImpact = result.topFeatures.first.impact.abs();
                final barWidth = maxImpact > 0 ? (absImpact / maxImpact) : 0.0;
                final isPositive = f.impact > 0;
                return Padding(
                  padding: const EdgeInsets.only(bottom: 10),
                  child: Row(children: [
                    SizedBox(width: 50, child: Text(f.feature, style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w500, color: Color(0xFF6B7280)))),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Stack(children: [
                        Container(height: 18, decoration: BoxDecoration(color: const Color(0xFFF3F4F6), borderRadius: BorderRadius.circular(4))),
                        FractionallySizedBox(widthFactor: barWidth.clamp(0.0, 1.0),
                          child: Container(height: 18, decoration: BoxDecoration(
                            color: isPositive ? const Color(0xFFEF4444).withOpacity(0.7) : const Color(0xFF2A9D8F).withOpacity(0.7),
                            borderRadius: BorderRadius.circular(4)))),
                      ]),
                    ),
                    const SizedBox(width: 8),
                    SizedBox(width: 55, child: Text(f.impact.toStringAsFixed(4), textAlign: TextAlign.end, style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: isPositive ? const Color(0xFFEF4444) : const Color(0xFF2A9D8F)))),
                  ]),
                );
              }),
              const SizedBox(height: 20),

              // ── AI Explanation ──────────────────────────────────
              const Text('AI Explanation', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: Color(0xFF1A1A2E))),
              const SizedBox(height: 8),
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(color: const Color(0xFFF5F7FA), borderRadius: BorderRadius.circular(12)),
                child: Text(result.aiExplanation, style: const TextStyle(fontSize: 14, height: 1.5, color: Color(0xFF374151))),
              ),
              const SizedBox(height: 20),

              // ── Ask Chatbot About This ─────────────────────────
              if (isTutorialActive)
                _BlinkingChatbotButton(onTap: onAskChatbot)
              else
                SizedBox(width: double.infinity, height: 50,
                  child: OutlinedButton.icon(
                    onPressed: onAskChatbot,
                    icon: const Icon(Icons.chat_bubble_outline, size: 20),
                    label: const Text('Ask Chatbot About This', style: TextStyle(fontSize: 15, fontWeight: FontWeight.w600)),
                    style: OutlinedButton.styleFrom(
                      foregroundColor: const Color(0xFF2A9D8F),
                      side: const BorderSide(color: Color(0xFF2A9D8F), width: 1.5),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
                    ),
                  ),
                ),
              const SizedBox(height: 16),
            ]),
          ),
        );
      },
        ),
      ),
    );
  }

  String _formatNow() {
    final now = DateTime.now();
    return '${now.day}/${now.month}/${now.year} ${now.hour.toString().padLeft(2, '0')}:${now.minute.toString().padLeft(2, '0')}';
  }
}

class _ResultRow extends StatelessWidget {
  final String label, value;
  const _ResultRow({required this.label, required this.value});
  @override
  Widget build(BuildContext context) {
    return Padding(padding: const EdgeInsets.only(bottom: 12), child: Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
      Text(label, style: const TextStyle(fontSize: 14, color: Color(0xFF6B7280))),
      Text(value, style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w700, color: Color(0xFF1A1A2E))),
    ]));
  }
}

class _LoadButton extends StatelessWidget {
  final String label;
  final IconData icon;
  final Color color;
  final bool isLoading;
  final VoidCallback onTap;

  const _LoadButton({required this.label, required this.icon, required this.color, required this.isLoading, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: isLoading ? null : onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 12),
        decoration: BoxDecoration(
          color: color.withOpacity(0.08),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: color.withOpacity(0.3)),
        ),
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          isLoading
              ? SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2, color: color))
              : Icon(icon, size: 20, color: color),
          const SizedBox(height: 6),
          Text(label, style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: color)),
        ]),
      ),
    );
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  Card Number Formatter — auto-adds space every 4 digits
// ══════════════════════════════════════════════════════════════════════════════

class _CardNumberFormatter extends TextInputFormatter {
  @override
  TextEditingValue formatEditUpdate(TextEditingValue oldValue, TextEditingValue newValue) {
    final digits = newValue.text.replaceAll(' ', '');
    if (digits.length > 16) return oldValue;

    final buffer = StringBuffer();
    for (int i = 0; i < digits.length; i++) {
      if (i > 0 && i % 4 == 0) buffer.write(' ');
      buffer.write(digits[i]);
    }
    final formatted = buffer.toString();
    return TextEditingValue(
      text: formatted,
      selection: TextSelection.collapsed(offset: formatted.length),
    );
  }
}

class _BlinkingChatbotButton extends StatefulWidget {
  final VoidCallback onTap;
  const _BlinkingChatbotButton({required this.onTap});
  @override
  State<_BlinkingChatbotButton> createState() => _BlinkingChatbotButtonState();
}

class _BlinkingChatbotButtonState extends State<_BlinkingChatbotButton> with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(vsync: this, duration: const Duration(milliseconds: 600))..repeat(reverse: true);
  }

  @override
  void dispose() { _controller.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        final color = Color.lerp(const Color(0xFF2A9D8F), const Color(0xFF38BDF8), _controller.value)!;
        return SizedBox(
          width: double.infinity,
          height: 50,
          child: GestureDetector(
            onTap: widget.onTap,
            child: Container(
              decoration: BoxDecoration(
                color: color.withOpacity(0.1 + _controller.value * 0.1),
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: color, width: 2),
                boxShadow: [
                  BoxShadow(color: color.withOpacity(0.3 + _controller.value * 0.3), blurRadius: 12, spreadRadius: 1),
                ],
              ),
              child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                Icon(Icons.chat_bubble_outline, size: 20, color: color),
                const SizedBox(width: 8),
                Text('Ask Chatbot About This', style: TextStyle(fontSize: 15, fontWeight: FontWeight.w700, color: color)),
              ]),
            ),
          ),
        );
      },
    );
  }
}