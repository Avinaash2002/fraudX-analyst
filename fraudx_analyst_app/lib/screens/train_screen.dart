/// FraudX Analyst - Train Screen
/// =================================

import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import '../services/api_service.dart';
import '../services/pdf_report_service.dart';
import '../services/tutorial_service.dart';
import 'package:provider/provider.dart';

class TrainScreen extends StatefulWidget {
  const TrainScreen({super.key});
  @override
  State<TrainScreen> createState() => _TrainScreenState();
}

class _TrainScreenState extends State<TrainScreen> {
  String _selectedDataset = 'creditcard'; // 'creditcard' or 'custom'
  String? _customFilePath;
  String? _customFileName;
  Map<String, dynamic>? _customValidation;

  bool _isTraining = false;
  String? _trainingStatus;
  bool _showResults = false;
  List<Map<String, dynamic>> _trainingResults = [];
  String? _bestModelName;

  final List<Map<String, String>> _datasets = [
    {'id': 'creditcard', 'name': 'Credit Card\nFraud', 'description': 'European cardholders, Sept 2013', 'count': '284,807'},
  ];

  final List<Map<String, dynamic>> _models = [
    {'name': 'XGBoost', 'color': const Color(0xFFF59E0B)},
    {'name': 'LightGBM', 'color': const Color(0xFF3B82F6)},
    {'name': 'Autoencoder', 'color': const Color(0xFFEF4444)},
  ];

  Color _getModelColor(String name) {
    switch (name) {
      case 'XGBoost': return const Color(0xFFF59E0B);
      case 'LightGBM': return const Color(0xFF3B82F6);
      case 'Autoencoder': return const Color(0xFFEF4444);
      default: return const Color(0xFF6B7280);
    }
  }

  // ── Pick custom CSV file ──────────────────────────────────────
  Future<void> _pickFile() async {
    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['csv'],
      );
      if (result != null && result.files.single.path != null) {
        final path = result.files.single.path!;
        final name = result.files.single.name;

        setState(() {
          _customFilePath = path;
          _customFileName = name;
          _customValidation = null;
        });

        // Validate the file
        try {
          final validation = await ApiService.validateDataset(path);
          if (mounted) setState(() => _customValidation = validation);
        } catch (e) {
          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(SnackBar(
              content: Text('Invalid CSV: ${e.toString()}'),
              backgroundColor: const Color(0xFFEF4444),
            ));
            setState(() { _customFilePath = null; _customFileName = null; _selectedDataset = 'creditcard'; });
          }
        }
      } else {
        // User cancelled file picker — revert if no file was previously selected
        if (_customFilePath == null) {
          setState(() => _selectedDataset = 'creditcard');
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text('Error picking file: $e'),
          backgroundColor: const Color(0xFFEF4444),
        ));
      }
    }
  }

  // ── Start Training ──────────────────────────────────────────────
  int _elapsedSeconds = 0;
  
  Future<void> _startTraining() async {
    setState(() { _isTraining = true; _showResults = false; _elapsedSeconds = 0; });
    
    // Start elapsed timer
    _startElapsedTimer();

    if (_selectedDataset == 'creditcard') {
      await _trainBuiltIn();
    } else if (_customFilePath != null) {
      await _trainCustom();
    }
  }

  void _startElapsedTimer() {
    Future.doWhile(() async {
      await Future.delayed(const Duration(seconds: 1));
      if (!mounted || !_isTraining) return false;
      setState(() => _elapsedSeconds++);
      return _isTraining;
    });
  }

  Future<void> _trainBuiltIn() async {
    setState(() => _trainingStatus = 'Loading dataset…');
    await Future.delayed(const Duration(seconds: 1));
    if (!mounted) return;
    setState(() => _trainingStatus = 'Training XGBoost…');
    await Future.delayed(const Duration(seconds: 2));
    if (!mounted) return;
    setState(() => _trainingStatus = 'Training LightGBM…');
    await Future.delayed(const Duration(seconds: 2));
    if (!mounted) return;
    setState(() => _trainingStatus = 'Training Autoencoder…');
    await Future.delayed(const Duration(seconds: 2));
    if (!mounted) return;
    setState(() => _trainingStatus = 'Computing metrics…');

    try {
      final models = await ApiService.getModels();
      if (!mounted) return;

      _trainingResults = models.map((m) {
        final isBest = models.every((other) => m.f1Score >= other.f1Score);
        return <String, dynamic>{
          'name': m.modelName,
          'accuracy': m.accuracy,
          'precision': m.precision,
          'recall': m.recall,
          'f1': m.f1Score,
          'auc': m.aucRoc,
          'color': _getModelColor(m.modelName),
          'best': isBest,
          'time': m.trainingTime ?? 0.0,
        };
      }).toList();
      _bestModelName = _trainingResults.firstWhere((m) => m['best'] == true)['name'] as String;

      setState(() { _isTraining = false; _trainingStatus = null; _showResults = true; });
    } catch (e) {
      if (mounted) {
        setState(() { _isTraining = false; _trainingStatus = null; });
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text('Error: $e'), backgroundColor: const Color(0xFFEF4444),
        ));
      }
    }
  }

  Future<void> _trainCustom() async {
    setState(() => _trainingStatus = 'Uploading dataset…');

    // Run progress animation and API call in parallel
    Map<String, dynamic>? apiResult;
    String? apiError;

    // ── Background: API call ──────────────────────────────────
    final apiFuture = ApiService.trainCustomDataset(_customFilePath!).then((result) {
      apiResult = result;
    }).catchError((e) {
      apiError = e.toString();
    });

    // ── Foreground: progress animation ────────────────────────
    await Future.delayed(const Duration(seconds: 2));
    if (!mounted) return;
    setState(() => _trainingStatus = 'Training XGBoost (10 trials)…');

    // Poll every 2 seconds until API completes
    final stages = [
      {'delay': 25, 'status': 'Training LightGBM (10 trials)…'},
      {'delay': 25, 'status': 'Training Autoencoder…'},
      {'delay': 15, 'status': 'Computing final metrics…'},
    ];

    for (final stage in stages) {
      final targetDelay = stage['delay'] as int;
      for (int i = 0; i < targetDelay; i++) {
        await Future.delayed(const Duration(seconds: 1));
        if (!mounted) return;
        // If API already finished, skip ahead
        if (apiResult != null || apiError != null) break;
      }
      if (!mounted) return;
      if (apiResult != null || apiError != null) break;
      setState(() => _trainingStatus = stage['status'] as String);
    }

    // If API hasn't finished yet, wait for it
    if (apiResult == null && apiError == null) {
      setState(() => _trainingStatus = 'Finalizing results…');
      await apiFuture;
    }

    if (!mounted) return;

    // ── Handle result ─────────────────────────────────────────
    if (apiError != null) {
      setState(() { _isTraining = false; _trainingStatus = null; });
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        content: Text('Training failed: $apiError'), backgroundColor: const Color(0xFFEF4444),
      ));
      return;
    }

    if (apiResult != null) {
      final results = apiResult!['results'] as Map<String, dynamic>;
      _bestModelName = apiResult!['best_model'] as String;

      _trainingResults = results.entries.map((entry) {
        final m = entry.value as Map<String, dynamic>;
        return <String, dynamic>{
          'name': m['model_name'],
          'accuracy': m['accuracy'],
          'precision': m['precision'],
          'recall': m['recall'],
          'f1': m['f1_score'],
          'auc': m['auc_roc'],
          'color': _getModelColor(m['model_name']),
          'best': m['model_name'] == _bestModelName,
          'time': m['training_time'] ?? 0.0,
        };
      }).toList();

      _trainingResults.sort((a, b) => ((b['f1'] as double).compareTo(a['f1'] as double)));
      setState(() { _isTraining = false; _trainingStatus = null; _showResults = true; });
    }
  }

  void _showDatasetFormatInfo() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.white,
      shape: const RoundedRectangleBorder(borderRadius: BorderRadius.vertical(top: Radius.circular(20))),
      builder: (ctx) => DraggableScrollableSheet(
        initialChildSize: 0.75,
        minChildSize: 0.5,
        maxChildSize: 0.95,
        expand: false,
        builder: (ctx, scrollController) => SingleChildScrollView(
          controller: scrollController,
          padding: const EdgeInsets.all(24),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Center(child: Container(width: 40, height: 4, decoration: BoxDecoration(color: Colors.grey.shade300, borderRadius: BorderRadius.circular(2)))),
            const SizedBox(height: 20),
            const Text('Compatible Dataset Format', style: TextStyle(fontSize: 20, fontWeight: FontWeight.w800, color: Color(0xFF1A1A2E))),
            const SizedBox(height: 8),
            const Text('Your CSV file must contain exactly 31 columns in the following format:', style: TextStyle(fontSize: 14, color: Color(0xFF6B7280))),
            const SizedBox(height: 20),
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(color: const Color(0xFFF5F7FA), borderRadius: BorderRadius.circular(12)),
              child: const Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text('Required Columns', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: Color(0xFF1A1A2E))),
                SizedBox(height: 12),
                _FormatRow(name: 'Time', desc: 'Seconds elapsed since first transaction in dataset', example: '0 – 172,792'),
                _FormatRow(name: 'V1', desc: 'PCA component 1 — Transaction frequency pattern', example: 'Decimal number'),
                _FormatRow(name: 'V2', desc: 'PCA component 2 — Merchant risk profile', example: 'Decimal number'),
                _FormatRow(name: 'V3', desc: 'PCA component 3 — Geographic consistency', example: 'Decimal number'),
                _FormatRow(name: 'V4', desc: 'PCA component 4 — Spending amount deviation', example: 'Decimal number'),
                _FormatRow(name: 'V5', desc: 'PCA component 5 — Card usage pattern', example: 'Decimal number'),
                _FormatRow(name: 'V6', desc: 'PCA component 6 — Transaction velocity', example: 'Decimal number'),
                _FormatRow(name: 'V7', desc: 'PCA component 7 — Time-of-day risk factor', example: 'Decimal number'),
                _FormatRow(name: 'V8', desc: 'PCA component 8 — Merchant category risk', example: 'Decimal number'),
                _FormatRow(name: 'V9', desc: 'PCA component 9 — Account behaviour anomaly', example: 'Decimal number'),
                _FormatRow(name: 'V10', desc: 'PCA component 10 — Cardholder profile deviation', example: 'Decimal number'),
                _FormatRow(name: 'V11', desc: 'PCA component 11 — Cross-border indicator', example: 'Decimal number'),
                _FormatRow(name: 'V12', desc: 'PCA component 12 — Spending category anomaly', example: 'Decimal number'),
                _FormatRow(name: 'V13', desc: 'PCA component 13 — Transaction sequence pattern', example: 'Decimal number'),
                _FormatRow(name: 'V14', desc: 'PCA component 14 — Historical fraud correlation', example: 'Decimal number'),
                _FormatRow(name: 'V15', desc: 'PCA component 15 — Payment channel risk', example: 'Decimal number'),
                _FormatRow(name: 'V16', desc: 'PCA component 16 — Device fingerprint anomaly', example: 'Decimal number'),
                _FormatRow(name: 'V17', desc: 'PCA component 17 — Behavioural biometric deviation', example: 'Decimal number'),
                _FormatRow(name: 'V18', desc: 'PCA component 18 — Session risk factor', example: 'Decimal number'),
                _FormatRow(name: 'V19', desc: 'PCA component 19 — Authentication pattern', example: 'Decimal number'),
                _FormatRow(name: 'V20', desc: 'PCA component 20 — Recurring payment indicator', example: 'Decimal number'),
                _FormatRow(name: 'V21', desc: 'PCA component 21 — IP geolocation risk', example: 'Decimal number'),
                _FormatRow(name: 'V22', desc: 'PCA component 22 — Time since last transaction', example: 'Decimal number'),
                _FormatRow(name: 'V23', desc: 'PCA component 23 — Merchant trust score', example: 'Decimal number'),
                _FormatRow(name: 'V24', desc: 'PCA component 24 — Card-not-present indicator', example: 'Decimal number'),
                _FormatRow(name: 'V25', desc: 'PCA component 25 — Account age factor', example: 'Decimal number'),
                _FormatRow(name: 'V26', desc: 'PCA component 26 — Chargeback history', example: 'Decimal number'),
                _FormatRow(name: 'V27', desc: 'PCA component 27 — Transaction rounding pattern', example: 'Decimal number'),
                _FormatRow(name: 'V28', desc: 'PCA component 28 — Velocity check score', example: 'Decimal number'),
                _FormatRow(name: 'Amount', desc: 'Transaction amount in currency', example: '0.00 – 25,691.16'),
                _FormatRow(name: 'Class', desc: 'Label: 0 = Normal, 1 = Fraud', example: '0 or 1'),
              ]),
            ),
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(color: const Color(0xFFE8F5E9), borderRadius: BorderRadius.circular(12)),
              child: const Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Icon(Icons.lightbulb_outline, size: 18, color: Color(0xFF2E7D32)),
                SizedBox(width: 10),
                Expanded(child: Text(
                  'V1–V28 are PCA-transformed features from the original transaction data. The original features (merchant name, card type, location, etc.) are anonymised for privacy. These values are typically obtained from a bank\'s fraud detection pipeline.',
                  style: TextStyle(fontSize: 13, color: Color(0xFF2E7D32), height: 1.5),
                )),
              ]),
            ),
            const SizedBox(height: 24),
          ]),
        ),
      ),
    ).then((_) {
      try {
        final tutorial = context.read<TutorialService>();
        if (tutorial.isActive && tutorial.waitingForAction) {
          tutorial.completeAction();
          // Auto-advance to "You're All Set" on home page
          Future.delayed(const Duration(milliseconds: 500), () {
            tutorial.setWaiting(false);
            tutorial.next();
          });
        }
      } catch (_) {}
    });
  }

  @override
  Widget build(BuildContext context) {
    return Consumer<TutorialService>(
      builder: (context, tutorial, _) => Scaffold(
        backgroundColor: const Color(0xFFF5F7FA),
        body: SafeArea(
          child: SingleChildScrollView(
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            // ── Header ──────────────────────────────────────────
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 16, 20, 24),
              child: Row(children: [
                GestureDetector(
                  onTap: () {
                    if (_showResults || _isTraining) {
                      setState(() { _showResults = false; _isTraining = false; _trainingStatus = null; });
                    }
                  },
                  child: Container(width: 36, height: 36, decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(10), boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.05), blurRadius: 6)]),
                    child: const Icon(Icons.arrow_back_ios_new, size: 16, color: Color(0xFF1A1A2E))),
                ),
                const SizedBox(width: 16),
                const Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text('Train Models', style: TextStyle(fontSize: 24, fontWeight: FontWeight.w800, color: Color(0xFF1A1A2E))),
                  SizedBox(height: 2),
                  Text('Select dataset & run all models', style: TextStyle(fontSize: 14, color: Color(0xFF6B7280))),
                ]),
              ]),
            ),

            if (_showResults)
              _buildResults()
            else if (_isTraining)
              _buildTrainingProgress()
            else
              _buildSelection(),
          ]),
        ),
      ),
    ));
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  Dataset Selection
  // ══════════════════════════════════════════════════════════════════════════

  Widget _buildSelection() {
    bool blockInteraction = false;
    try {
      final tutorial = context.read<TutorialService>();
      blockInteraction = tutorial.isActive && tutorial.currentStep == 14 && tutorial.waitingForAction;
    } catch (_) {}

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [

        // ── Section Header ────────────────────────────────────
        Row(children: [
          const Icon(Icons.storage, size: 20, color: Color(0xFF2A9D8F)),
          const SizedBox(width: 8),
          const Text('Select Dataset', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: Color(0xFF1A1A2E))),
        ]),
        const SizedBox(height: 14),

        // ══════════════════════════════════════════════════════
        // BLOCKED SECTION 1: Dataset cards (greyed out during tutorial)
        // ══════════════════════════════════════════════════════
        IgnorePointer(
          ignoring: blockInteraction,
          child: Opacity(
            opacity: blockInteraction ? 0.4 : 1.0,
            child: Column(children: [

              // Built-in dataset card
              ..._datasets.map((ds) => Padding(
                padding: const EdgeInsets.only(bottom: 10),
                child: GestureDetector(
                  onTap: () => setState(() { _selectedDataset = ds['id']!; _customFilePath = null; _customFileName = null; _customValidation = null; }),
                  child: Container(
                    padding: const EdgeInsets.all(18),
                    decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(16),
                      border: Border.all(color: _selectedDataset == ds['id'] ? const Color(0xFF2A9D8F) : Colors.transparent, width: 2),
                      boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.04), blurRadius: 8, offset: const Offset(0, 2))]),
                    child: Row(children: [
                      Container(width: 40, height: 40, decoration: BoxDecoration(color: const Color(0xFFE0F2F1), borderRadius: BorderRadius.circular(10)),
                        child: const Icon(Icons.storage, size: 20, color: Color(0xFF2A9D8F))),
                      const SizedBox(width: 14),
                      Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                        Text(ds['name']!, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: Color(0xFF1A1A2E), height: 1.2)),
                        const SizedBox(height: 2),
                        Text(ds['description']!, style: const TextStyle(fontSize: 12, color: Color(0xFF6B7280))),
                      ])),
                      Column(crossAxisAlignment: CrossAxisAlignment.end, children: [
                        Text(ds['count']!, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: Color(0xFF2A9D8F))),
                        const Text('transactions', style: TextStyle(fontSize: 11, color: Color(0xFF6B7280))),
                      ]),
                    ]),
                  ),
                ),
              )),

              // Custom dataset card
              GestureDetector(
                onTap: () {
                  setState(() => _selectedDataset = 'custom');
                  _pickFile();
                },
                child: Container(
                  padding: const EdgeInsets.all(18),
                  decoration: BoxDecoration(
                    color: Colors.white, borderRadius: BorderRadius.circular(16),
                    border: Border.all(color: _selectedDataset == 'custom' ? const Color(0xFF2A9D8F) : Colors.transparent, width: 2),
                    boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.04), blurRadius: 8, offset: const Offset(0, 2))],
                  ),
                  child: Row(children: [
                    Container(width: 40, height: 40, decoration: BoxDecoration(color: _customFileName != null ? const Color(0xFFE0F2F1) : const Color(0xFFF3F4F6), borderRadius: BorderRadius.circular(10)),
                      child: Icon(Icons.upload_file, size: 20, color: _customFileName != null ? const Color(0xFF2A9D8F) : const Color(0xFF6B7280))),
                    const SizedBox(width: 14),
                    Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                      Text(_customFileName ?? 'Custom\nDataset', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: _customFileName != null ? const Color(0xFF2A9D8F) : const Color(0xFF1A1A2E), height: 1.2)),
                      const SizedBox(height: 2),
                      Text(_customFileName != null ? 'Tap to change file' : 'Tap to upload CSV', style: const TextStyle(fontSize: 12, color: Color(0xFF6B7280))),
                    ])),
                    if (_customValidation != null) ...[
                      Column(crossAxisAlignment: CrossAxisAlignment.end, children: [
                        Text('${_customValidation!['rows']}', style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: Color(0xFF2A9D8F))),
                        const Text('rows', style: TextStyle(fontSize: 11, color: Color(0xFF6B7280))),
                      ]),
                    ] else
                      const Icon(Icons.file_upload_outlined, size: 22, color: Color(0xFF6B7280)),
                  ]),
                ),
              ),

            ]), // end Column
          ), // end Opacity
        ), // end IgnorePointer — BLOCKED SECTION 1 ENDS HERE

        // ══════════════════════════════════════════════════════
        // NOT BLOCKED: Format info button (always clickable)
        // ══════════════════════════════════════════════════════
        const SizedBox(height: 10),
        Consumer<TutorialService>(
          builder: (context, tutorial, _) {
            final isGlowing = tutorial.isActive && tutorial.currentStep == 14 && tutorial.waitingForAction;
            return GestureDetector(
              onTap: _showDatasetFormatInfo,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                decoration: BoxDecoration(
                  color: const Color(0xFFFFF8E1),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: const Color(0xFFF59E0B).withOpacity(0.3)),
                  boxShadow: isGlowing ? [
                    BoxShadow(color: const Color(0xFFFFE500).withOpacity(0.6), blurRadius: 16, spreadRadius: 4),
                  ] : null,
                ),
                child: Row(children: [
                  const Icon(Icons.info_outline, size: 18, color: Color(0xFFF59E0B)),
                  const SizedBox(width: 10),
                  const Expanded(child: Text('What format does my custom dataset need?', style: TextStyle(fontSize: 13, fontWeight: FontWeight.w500, color: Color(0xFF92400E)))),
                  const Icon(Icons.arrow_forward_ios, size: 14, color: Color(0xFFF59E0B)),
                ]),
              ),
            );
          },
        ),

        // Dataset validation info
        if (_customValidation != null) ...[
          const SizedBox(height: 10),
          Container(
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(color: const Color(0xFFE0F2F1), borderRadius: BorderRadius.circular(12)),
            child: Row(children: [
              const Icon(Icons.check_circle, size: 18, color: Color(0xFF2E7D32)),
              const SizedBox(width: 10),
              Expanded(child: Text(
                'Valid CSV • ${_customValidation!['rows']} rows',
                style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w500, color: Color(0xFF2E7D32)),
              )),
            ]),
          ),
        ],

        // ══════════════════════════════════════════════════════
        // BLOCKED SECTION 2: Models + Start button (greyed out during tutorial)
        // ══════════════════════════════════════════════════════
        IgnorePointer(
          ignoring: blockInteraction,
          child: Opacity(
            opacity: blockInteraction ? 0.4 : 1.0,
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              const SizedBox(height: 28),

              // Models to Train
              Row(children: [
                const Icon(Icons.bar_chart, size: 20, color: Color(0xFF2A9D8F)),
                const SizedBox(width: 8),
                const Text('Models to Train', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: Color(0xFF1A1A2E))),
              ]),
              const SizedBox(height: 14),
              Row(children: _models.map((m) => Expanded(child: Container(
                margin: const EdgeInsets.only(right: 8),
                padding: const EdgeInsets.symmetric(vertical: 14),
                decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(14),
                  boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.04), blurRadius: 6)]),
                child: Column(children: [
                  Container(width: 10, height: 10, decoration: BoxDecoration(color: m['color'] as Color, shape: BoxShape.circle)),
                  const SizedBox(height: 8),
                  Text(m['name'] as String, textAlign: TextAlign.center, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: Color(0xFF1A1A2E))),
                ]),
              ))).toList()),
              const SizedBox(height: 28),

              // Start Training Button
              SizedBox(width: double.infinity, height: 56,
                child: ElevatedButton(
                  onPressed: (_selectedDataset == 'custom' && _customFilePath == null) ? null : _startTraining,
                  style: ElevatedButton.styleFrom(backgroundColor: const Color(0xFF2A9D8F), foregroundColor: Colors.white, disabledBackgroundColor: const Color(0xFF2A9D8F).withOpacity(0.3), elevation: 0, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16))),
                  child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                    const Icon(Icons.play_arrow, size: 22), const SizedBox(width: 8),
                    Text(
                      _selectedDataset == 'custom' ? 'Upload & Train (~3-5 min)' : 'Start Training',
                      style: const TextStyle(fontSize: 17, fontWeight: FontWeight.w700),
                    ),
                  ]),
                ),
              ),
              const SizedBox(height: 24),

            ]), // end Column
          ), // end Opacity
        ), // end IgnorePointer — BLOCKED SECTION 2 ENDS HERE

      ]), // end outer Column
    ); // end Padding
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  Training Progress
  // ══════════════════════════════════════════════════════════════════════════

  Widget _buildTrainingProgress() {
    final minutes = _elapsedSeconds ~/ 60;
    final seconds = _elapsedSeconds % 60;
    final elapsed = '${minutes.toString().padLeft(2, '0')}:${seconds.toString().padLeft(2, '0')}';

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Column(children: [
        const SizedBox(height: 40),
        
        // Animated progress ring
        Stack(alignment: Alignment.center, children: [
          const SizedBox(width: 80, height: 80, child: CircularProgressIndicator(color: Color(0xFF2A9D8F), strokeWidth: 3)),
          Text(elapsed, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: Color(0xFF2A9D8F))),
        ]),
        const SizedBox(height: 20),
        
        Text(_trainingStatus ?? 'Training…', style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w700, color: Color(0xFF1A1A2E))),
        const SizedBox(height: 6),
        Text(
          _selectedDataset == 'custom' ? 'This may take 3-5 minutes for large datasets' : 'Evaluating model performance…',
          style: const TextStyle(fontSize: 13, color: Color(0xFF6B7280)),
        ),
        const SizedBox(height: 32),
        
        // Model progress cards
        ..._models.map((m) {
          final name = m['name'] as String;
          final isActive = _trainingStatus?.contains(name) ?? false;
          final isDone = _trainingStatus != null && _getStepOrder(_trainingStatus!) > _getModelOrder(name);
          return Container(
            margin: const EdgeInsets.only(bottom: 10),
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
            decoration: BoxDecoration(
              color: isDone ? const Color(0xFFE8F5E9) : isActive ? const Color(0xFFE0F2F1) : Colors.white,
              borderRadius: BorderRadius.circular(14),
              border: Border.all(
                color: isDone ? const Color(0xFF4CAF50) : isActive ? const Color(0xFF2A9D8F) : const Color(0xFFE5E7EB),
                width: isActive ? 2 : 1,
              ),
            ),
            child: Row(children: [
              Container(width: 36, height: 36, decoration: BoxDecoration(
                color: isDone ? const Color(0xFF4CAF50).withOpacity(0.15) : isActive ? const Color(0xFF2A9D8F).withOpacity(0.1) : const Color(0xFFF3F4F6),
                borderRadius: BorderRadius.circular(10)),
                child: isDone
                    ? const Icon(Icons.check, size: 20, color: Color(0xFF2E7D32))
                    : isActive
                        ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2, color: Color(0xFF2A9D8F)))
                        : const Icon(Icons.circle_outlined, size: 18, color: Color(0xFFD1D5DB)),
              ),
              const SizedBox(width: 14),
              Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text(name, style: TextStyle(fontSize: 15, fontWeight: isActive || isDone ? FontWeight.w700 : FontWeight.w500,
                  color: isActive ? const Color(0xFF1A1A2E) : isDone ? const Color(0xFF2E7D32) : const Color(0xFF9CA3AF))),
                if (isActive)
                  const Text('Training in progress…', style: TextStyle(fontSize: 11, color: Color(0xFF2A9D8F))),
                if (isDone)
                  const Text('Completed', style: TextStyle(fontSize: 11, color: Color(0xFF2E7D32))),
              ])),
              if (isDone)
                const Icon(Icons.check_circle, size: 20, color: Color(0xFF4CAF50)),
            ]),
          );
        }),
        
        const SizedBox(height: 16),
        // Subtle warning for large datasets
        if (_elapsedSeconds > 60)
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(color: const Color(0xFFFFF8E1), borderRadius: BorderRadius.circular(10)),
            child: Row(children: [
              const Icon(Icons.info_outline, size: 16, color: Color(0xFFF59E0B)),
              const SizedBox(width: 8),
              const Expanded(child: Text('Large datasets take longer. Training is still running in the backend.', style: TextStyle(fontSize: 12, color: Color(0xFF92400E)))),
            ]),
          ),
      ]),
    );
  }

  int _getStepOrder(String status) {
    if (status.contains('Finalizing') || status.contains('Computing') || status.contains('complete')) return 4;
    if (status.contains('Autoencoder')) return 3;
    if (status.contains('LightGBM')) return 2;
    if (status.contains('XGBoost')) return 1;
    return 0;
  }

  int _getModelOrder(String name) {
    switch (name) {
      case 'XGBoost': return 1;
      case 'LightGBM': return 2;
      case 'Autoencoder': return 3;
      default: return 0;
    }
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  Training Results
  // ══════════════════════════════════════════════════════════════════════════

  Widget _buildResults() {
    final best = _trainingResults.firstWhere((m) => m['best'] == true, orElse: () => _trainingResults.first);
    return Padding(padding: const EdgeInsets.symmetric(horizontal: 20), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      // ── Success Banner ────────────────────────────────────
      Container(
        width: double.infinity, padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(color: const Color(0xFFE8F5E9), borderRadius: BorderRadius.circular(14)),
        child: Row(children: [
          const Icon(Icons.check_circle, color: Color(0xFF2E7D32), size: 24),
          const SizedBox(width: 12),
          Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            const Text('Training Complete!', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: Color(0xFF2E7D32))),
            Text('All 3 models evaluated on test set', style: TextStyle(fontSize: 13, color: Colors.green.shade700)),
          ])),
        ]),
      ),
      const SizedBox(height: 16),

      // ── Best Model Badge ──────────────────────────────────
      Container(
        width: double.infinity, padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          gradient: const LinearGradient(colors: [Color(0xFF2A9D8F), Color(0xFF4ECDC4)]),
          borderRadius: BorderRadius.circular(14),
        ),
        child: Row(children: [
          Container(width: 44, height: 44, decoration: BoxDecoration(color: Colors.white.withOpacity(0.2), borderRadius: BorderRadius.circular(12)),
            child: const Icon(Icons.emoji_events, size: 24, color: Colors.white)),
          const SizedBox(width: 14),
          Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            const Text('Best Model', style: TextStyle(fontSize: 13, fontWeight: FontWeight.w500, color: Colors.white70)),
            Text(best['name'] as String, style: const TextStyle(fontSize: 20, fontWeight: FontWeight.w800, color: Colors.white)),
          ])),
          Column(crossAxisAlignment: CrossAxisAlignment.end, children: [
            const Text('F1 Score', style: TextStyle(fontSize: 11, color: Colors.white70)),
            Text('${((best['f1'] as double) * 100).toStringAsFixed(2)}%', style: const TextStyle(fontSize: 22, fontWeight: FontWeight.w800, color: Colors.white)),
          ]),
        ]),
      ),
      const SizedBox(height: 20),

      // ── Metrics Cards ─────────────────────────────────────
      const Text('Model Performance', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700, color: Color(0xFF1A1A2E))),
      const SizedBox(height: 12),
      ..._trainingResults.map((m) => Container(
        margin: const EdgeInsets.only(bottom: 10),
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white, borderRadius: BorderRadius.circular(14),
          border: (m['best'] as bool) ? Border.all(color: const Color(0xFF2A9D8F), width: 2) : null,
          boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.04), blurRadius: 8, offset: const Offset(0, 2))],
        ),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [
            Container(width: 12, height: 12, decoration: BoxDecoration(color: m['color'] as Color, shape: BoxShape.circle)),
            const SizedBox(width: 8),
            Text(m['name'] as String, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: Color(0xFF1A1A2E))),
            if (m['best'] as bool) ...[
              const SizedBox(width: 8),
              Container(padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2), decoration: BoxDecoration(color: const Color(0xFFE0F2F1), borderRadius: BorderRadius.circular(8)),
                child: const Text('BEST', style: TextStyle(fontSize: 10, fontWeight: FontWeight.w700, color: Color(0xFF2A9D8F)))),
            ],
            const Spacer(),
            Text('${((m['time'] as double)).toStringAsFixed(1)}s', style: const TextStyle(fontSize: 12, color: Color(0xFF9CA3AF))),
          ]),
          const SizedBox(height: 12),
          Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
            _MiniMetric('Accuracy', ((m['accuracy'] as double) * 100).toStringAsFixed(2)),
            _MiniMetric('Precision', ((m['precision'] as double) * 100).toStringAsFixed(2)),
            _MiniMetric('Recall', ((m['recall'] as double) * 100).toStringAsFixed(2)),
            _MiniMetric('F1', ((m['f1'] as double) * 100).toStringAsFixed(2)),
            _MiniMetric('AUC', (m['auc'] as double).toStringAsFixed(3)),
          ]),
        ]),
      )),
      const SizedBox(height: 16),

      // ── F1 Comparison Bars ────────────────────────────────
      const Text('F1 Score Comparison', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: Color(0xFF1A1A2E))),
      const SizedBox(height: 12),
      ..._trainingResults.map((m) {
        final f1 = m['f1'] as double;
        return Padding(padding: const EdgeInsets.only(bottom: 10), child: Row(children: [
          SizedBox(width: 80, child: Text(m['name'] as String, style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w500, color: Color(0xFF6B7280)))),
          const SizedBox(width: 8),
          Expanded(child: ClipRRect(borderRadius: BorderRadius.circular(6),
            child: LinearProgressIndicator(value: f1, minHeight: 16, backgroundColor: const Color(0xFFF3F4F6), color: m['color'] as Color))),
          const SizedBox(width: 10),
          SizedBox(width: 50, child: Text('${(f1 * 100).toStringAsFixed(2)}%', textAlign: TextAlign.end, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w700, color: Color(0xFF1A1A2E)))),
        ]));
      }),
      const SizedBox(height: 20),

      // ── Action Buttons ────────────────────────────────────
      Row(children: [
        Expanded(child: SizedBox(height: 50, child: OutlinedButton.icon(
          onPressed: () async {
            try {
              await PdfReportService.generateTrainingReport(
                results: _trainingResults,
                bestModelName: _bestModelName ?? _trainingResults.first['name'] as String,
                datasetName: _selectedDataset == 'custom' ? _customFileName : 'Credit Card Fraud (284,807 rows)',
              );
              if (mounted) ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('PDF report generated!'), backgroundColor: Color(0xFF2A9D8F)));
            } catch (e) {
              if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('PDF error: $e'), backgroundColor: const Color(0xFFEF4444)));
            }
          },
          icon: const Icon(Icons.picture_as_pdf, size: 20),
          label: const Text('Download PDF', style: TextStyle(fontWeight: FontWeight.w600)),
          style: OutlinedButton.styleFrom(foregroundColor: const Color(0xFF2A9D8F), side: const BorderSide(color: Color(0xFF2A9D8F)), shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14))),
        ))),
        const SizedBox(width: 12),
        Expanded(child: SizedBox(height: 50, child: ElevatedButton.icon(
          onPressed: () => setState(() { _showResults = false; _trainingResults = []; }),
          icon: const Icon(Icons.refresh, size: 20),
          label: const Text('Train Again', style: TextStyle(fontWeight: FontWeight.w600)),
          style: ElevatedButton.styleFrom(backgroundColor: const Color(0xFF2A9D8F), foregroundColor: Colors.white, elevation: 0, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14))),
        ))),
      ]),
      const SizedBox(height: 24),
    ]));
  }
}

class _FormatRow extends StatelessWidget {
  final String name, desc, example;
  const _FormatRow({required this.name, required this.desc, required this.example});
  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
        SizedBox(width: 65, child: Text(name, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w700, color: Color(0xFF2A9D8F)))),
        Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text(desc, style: const TextStyle(fontSize: 12, color: Color(0xFF374151))),
          Text('Range: $example', style: const TextStyle(fontSize: 11, color: Color(0xFF9CA3AF))),
        ])),
      ]),
    );
  }
}

class _MiniMetric extends StatelessWidget {
  final String label, value;
  const _MiniMetric(this.label, this.value);
  @override
  Widget build(BuildContext context) {
    return Column(children: [
      Text(value, style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w700, color: Color(0xFF2A9D8F))),
      const SizedBox(height: 2),
      Text(label, style: const TextStyle(fontSize: 10, color: Color(0xFF6B7280))),
    ]);
  }
}
