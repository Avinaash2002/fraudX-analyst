/// FraudX Analyst - PDF Report Service
/// =======================================
/// Generates professional PDF reports for:
/// 1. Training results (from Train screen)
/// 2. Model comparison (from Models screen)

import 'dart:io';
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:path_provider/path_provider.dart';
import 'package:open_filex/open_filex.dart';

class PdfReportService {
  static const _teal = PdfColor.fromInt(0xFF2A9D8F);
  static const _dark = PdfColor.fromInt(0xFF1A1A2E);
  static const _gray = PdfColor.fromInt(0xFF6B7280);
  static const _lightBg = PdfColor.fromInt(0xFFF5F7FA);
  static const _green = PdfColor.fromInt(0xFF2E7D32);
  static const _red = PdfColor.fromInt(0xFFEF4444);

  // ══════════════════════════════════════════════════════════════════════════
  //  Training Results Report
  // ══════════════════════════════════════════════════════════════════════════

  static Future<void> generateTrainingReport({
    required List<Map<String, dynamic>> results,
    required String bestModelName,
    String? datasetName,
  }) async {
    final pdf = pw.Document();
    final now = DateTime.now();
    final dateStr = '${now.day}/${now.month}/${now.year} ${now.hour.toString().padLeft(2, '0')}:${now.minute.toString().padLeft(2, '0')}';

    pdf.addPage(
      pw.MultiPage(
        pageFormat: PdfPageFormat.a4,
        margin: const pw.EdgeInsets.all(40),
        header: (context) => _buildHeader('Training Report'),
        footer: (context) => _buildFooter(context),
        build: (context) => [
          // Title section
          pw.Container(
            width: double.infinity,
            padding: const pw.EdgeInsets.all(20),
            decoration: pw.BoxDecoration(
              color: _teal,
              borderRadius: pw.BorderRadius.circular(8),
            ),
            child: pw.Column(crossAxisAlignment: pw.CrossAxisAlignment.start, children: [
              pw.Text('FraudX Analyst', style: pw.TextStyle(fontSize: 24, fontWeight: pw.FontWeight.bold, color: PdfColors.white)),
              pw.SizedBox(height: 4),
              pw.Text('Model Training Results Report', style: const pw.TextStyle(fontSize: 14, color: PdfColors.white)),
              pw.SizedBox(height: 8),
              pw.Text('Generated: $dateStr', style: pw.TextStyle(fontSize: 10, color: PdfColors.white.shade(0.8))),
              if (datasetName != null)
                pw.Text('Dataset: $datasetName', style: pw.TextStyle(fontSize: 10, color: PdfColors.white.shade(0.8))),
            ]),
          ),
          pw.SizedBox(height: 20),

          // Best model highlight
          pw.Container(
            width: double.infinity,
            padding: const pw.EdgeInsets.all(16),
            decoration: pw.BoxDecoration(
              border: pw.Border.all(color: _teal, width: 2),
              borderRadius: pw.BorderRadius.circular(8),
            ),
            child: pw.Row(children: [
              pw.Container(
                width: 40, height: 40,
                decoration: pw.BoxDecoration(color: PdfColor.fromInt(0xFFE0F2F1), borderRadius: pw.BorderRadius.circular(8)),
                child: pw.Center(child: pw.Text('🏆', style: const pw.TextStyle(fontSize: 20))),
              ),
              pw.SizedBox(width: 14),
              pw.Expanded(child: pw.Column(crossAxisAlignment: pw.CrossAxisAlignment.start, children: [
                pw.Text('Best Performing Model', style: pw.TextStyle(fontSize: 10, color: _gray)),
                pw.Text(bestModelName, style: pw.TextStyle(fontSize: 18, fontWeight: pw.FontWeight.bold, color: _dark)),
              ])),
              pw.Column(crossAxisAlignment: pw.CrossAxisAlignment.end, children: [
                pw.Text('F1 Score', style: pw.TextStyle(fontSize: 10, color: _gray)),
                pw.Text(
                  '${((results.firstWhere((m) => m['name'] == bestModelName)['f1'] as double) * 100).toStringAsFixed(1)}%',
                  style: pw.TextStyle(fontSize: 18, fontWeight: pw.FontWeight.bold, color: _teal),
                ),
              ]),
            ]),
          ),
          pw.SizedBox(height: 24),

          // Metrics table
          pw.Text('Model Performance Comparison', style: pw.TextStyle(fontSize: 16, fontWeight: pw.FontWeight.bold, color: _dark)),
          pw.SizedBox(height: 12),
          _buildMetricsTable(results, bestModelName),
          pw.SizedBox(height: 24),

          // Individual model cards
          pw.Text('Detailed Results', style: pw.TextStyle(fontSize: 16, fontWeight: pw.FontWeight.bold, color: _dark)),
          pw.SizedBox(height: 12),
          ...results.map((m) => _buildModelCard(m, m['name'] == bestModelName)),
          pw.SizedBox(height: 20),

          // Methodology note
          pw.Container(
            width: double.infinity,
            padding: const pw.EdgeInsets.all(14),
            decoration: pw.BoxDecoration(color: _lightBg, borderRadius: pw.BorderRadius.circular(6)),
            child: pw.Column(crossAxisAlignment: pw.CrossAxisAlignment.start, children: [
              pw.Text('Methodology', style: pw.TextStyle(fontSize: 12, fontWeight: pw.FontWeight.bold, color: _dark)),
              pw.SizedBox(height: 6),
              pw.Text(
                '• Data split: 70% training / 15% validation / 15% test (stratified)\n'
                '• Scalers fit on training data only (no data leakage)\n'
                '• XGBoost & LightGBM: Optuna hyperparameter tuning\n'
                '• Autoencoder: trained on normal transactions only, threshold optimized on validation set\n'
                '• Final evaluation performed once on held-out test set',
                style: pw.TextStyle(fontSize: 9, color: _gray, lineSpacing: 3),
              ),
            ]),
          ),
        ],
      ),
    );

    await _saveAndOpen(pdf, 'FraudX_Training_Report');
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  Model Comparison Report
  // ══════════════════════════════════════════════════════════════════════════

  static Future<void> generateComparisonReport({
    required List<Map<String, dynamic>> models,
    required String bestModelName,
  }) async {
    final pdf = pw.Document();
    final now = DateTime.now();
    final dateStr = '${now.day}/${now.month}/${now.year} ${now.hour.toString().padLeft(2, '0')}:${now.minute.toString().padLeft(2, '0')}';

    pdf.addPage(
      pw.MultiPage(
        pageFormat: PdfPageFormat.a4,
        margin: const pw.EdgeInsets.all(40),
        header: (context) => _buildHeader('Comparison Report'),
        footer: (context) => _buildFooter(context),
        build: (context) => [
          // Title
          pw.Container(
            width: double.infinity,
            padding: const pw.EdgeInsets.all(20),
            decoration: pw.BoxDecoration(color: _teal, borderRadius: pw.BorderRadius.circular(8)),
            child: pw.Column(crossAxisAlignment: pw.CrossAxisAlignment.start, children: [
              pw.Text('FraudX Analyst', style: pw.TextStyle(fontSize: 24, fontWeight: pw.FontWeight.bold, color: PdfColors.white)),
              pw.SizedBox(height: 4),
              pw.Text('Model Comparison Report', style: const pw.TextStyle(fontSize: 14, color: PdfColors.white)),
              pw.SizedBox(height: 8),
              pw.Text('Generated: $dateStr', style: pw.TextStyle(fontSize: 10, color: PdfColors.white.shade(0.8))),
            ]),
          ),
          pw.SizedBox(height: 20),

          // Best model
          pw.Container(
            width: double.infinity,
            padding: const pw.EdgeInsets.all(16),
            decoration: pw.BoxDecoration(border: pw.Border.all(color: _teal, width: 2), borderRadius: pw.BorderRadius.circular(8)),
            child: pw.Row(children: [
              pw.Expanded(child: pw.Column(crossAxisAlignment: pw.CrossAxisAlignment.start, children: [
                pw.Text('Best Performing Model', style: pw.TextStyle(fontSize: 10, color: _gray)),
                pw.Text(bestModelName, style: pw.TextStyle(fontSize: 18, fontWeight: pw.FontWeight.bold, color: _dark)),
              ])),
              pw.Column(crossAxisAlignment: pw.CrossAxisAlignment.end, children: [
                pw.Text('F1 Score', style: pw.TextStyle(fontSize: 10, color: _gray)),
                pw.Text(
                  '${((models.firstWhere((m) => m['name'] == bestModelName)['f1'] as double) * 100).toStringAsFixed(1)}%',
                  style: pw.TextStyle(fontSize: 18, fontWeight: pw.FontWeight.bold, color: _teal),
                ),
              ]),
            ]),
          ),
          pw.SizedBox(height: 24),

          // Comparison table
          pw.Text('Side-by-Side Comparison', style: pw.TextStyle(fontSize: 16, fontWeight: pw.FontWeight.bold, color: _dark)),
          pw.SizedBox(height: 12),
          _buildMetricsTable(models, bestModelName),
          pw.SizedBox(height: 24),

          // Per-model details
          pw.Text('Model Details', style: pw.TextStyle(fontSize: 16, fontWeight: pw.FontWeight.bold, color: _dark)),
          pw.SizedBox(height: 12),
          ...models.map((m) => _buildModelCard(m, m['name'] == bestModelName)),
          pw.SizedBox(height: 20),

          // Metric explanations
          pw.Container(
            width: double.infinity,
            padding: const pw.EdgeInsets.all(14),
            decoration: pw.BoxDecoration(color: _lightBg, borderRadius: pw.BorderRadius.circular(6)),
            child: pw.Column(crossAxisAlignment: pw.CrossAxisAlignment.start, children: [
              pw.Text('Metric Definitions', style: pw.TextStyle(fontSize: 12, fontWeight: pw.FontWeight.bold, color: _dark)),
              pw.SizedBox(height: 6),
              pw.Text(
                '• Accuracy: Overall proportion of correct predictions\n'
                '• Precision: Of transactions flagged as fraud, how many truly are fraud\n'
                '• Recall: Of all actual fraud transactions, how many were detected\n'
                '• F1 Score: Harmonic mean of precision and recall (primary metric)\n'
                '• AUC-ROC: Model\'s ability to distinguish between fraud and normal across all thresholds',
                style: pw.TextStyle(fontSize: 9, color: _gray, lineSpacing: 3),
              ),
            ]),
          ),
        ],
      ),
    );

    await _saveAndOpen(pdf, 'FraudX_Comparison_Report');
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  Shared Components
  // ══════════════════════════════════════════════════════════════════════════

  static pw.Widget _buildHeader(String subtitle) {
    return pw.Container(
      margin: const pw.EdgeInsets.only(bottom: 16),
      child: pw.Row(mainAxisAlignment: pw.MainAxisAlignment.spaceBetween, children: [
        pw.Text('FraudX Analyst', style: pw.TextStyle(fontSize: 10, fontWeight: pw.FontWeight.bold, color: _teal)),
        pw.Text(subtitle, style: pw.TextStyle(fontSize: 10, color: _gray)),
      ]),
    );
  }

  static pw.Widget _buildFooter(pw.Context context) {
    return pw.Container(
      margin: const pw.EdgeInsets.only(top: 12),
      decoration: const pw.BoxDecoration(border: pw.Border(top: pw.BorderSide(color: PdfColors.grey300, width: 0.5))),
      padding: const pw.EdgeInsets.only(top: 8),
      child: pw.Row(mainAxisAlignment: pw.MainAxisAlignment.spaceBetween, children: [
        pw.Text('FraudX Analyst — Credit Card Fraud Detection with XAI', style: pw.TextStyle(fontSize: 8, color: _gray)),
        pw.Text('Page ${context.pageNumber} of ${context.pagesCount}', style: pw.TextStyle(fontSize: 8, color: _gray)),
      ]),
    );
  }

  static pw.Widget _buildMetricsTable(List<Map<String, dynamic>> results, String bestModel) {
    return pw.Table(
      border: pw.TableBorder.all(color: PdfColors.grey300, width: 0.5),
      columnWidths: {
        0: const pw.FlexColumnWidth(2),
        1: const pw.FlexColumnWidth(1.2),
        2: const pw.FlexColumnWidth(1.2),
        3: const pw.FlexColumnWidth(1.2),
        4: const pw.FlexColumnWidth(1.2),
        5: const pw.FlexColumnWidth(1.2),
      },
      children: [
        // Header row
        pw.TableRow(
          decoration: const pw.BoxDecoration(color: PdfColor.fromInt(0xFFE0F2F1)),
          children: ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']
              .map((h) => pw.Padding(
                    padding: const pw.EdgeInsets.all(8),
                    child: pw.Text(h, style: pw.TextStyle(fontSize: 9, fontWeight: pw.FontWeight.bold, color: _dark), textAlign: pw.TextAlign.center),
                  ))
              .toList(),
        ),
        // Data rows
        ...results.map((m) {
          final isBest = m['name'] == bestModel;
          return pw.TableRow(
            decoration: isBest ? const pw.BoxDecoration(color: PdfColor.fromInt(0xFFF0FDF9)) : null,
            children: [
              pw.Padding(
                padding: const pw.EdgeInsets.all(8),
                child: pw.Row(children: [
                  pw.Text(m['name'] as String, style: pw.TextStyle(fontSize: 9, fontWeight: isBest ? pw.FontWeight.bold : pw.FontWeight.normal, color: _dark)),
                  if (isBest) ...[
                    pw.SizedBox(width: 4),
                    pw.Container(
                      padding: const pw.EdgeInsets.symmetric(horizontal: 4, vertical: 1),
                      decoration: pw.BoxDecoration(color: _teal, borderRadius: pw.BorderRadius.circular(3)),
                      child: pw.Text('BEST', style: pw.TextStyle(fontSize: 6, fontWeight: pw.FontWeight.bold, color: PdfColors.white)),
                    ),
                  ],
                ]),
              ),
              _tableCell('${((m['accuracy'] as double) * 100).toStringAsFixed(2)}%'),
              _tableCell('${((m['precision'] as double) * 100).toStringAsFixed(2)}%'),
              _tableCell('${((m['recall'] as double) * 100).toStringAsFixed(2)}%'),
              _tableCell('${((m['f1'] as double) * 100).toStringAsFixed(2)}%', bold: isBest),
              _tableCell((m['auc'] as double).toStringAsFixed(4)),
            ],
          );
        }),
      ],
    );
  }

  static pw.Widget _tableCell(String text, {bool bold = false}) {
    return pw.Padding(
      padding: const pw.EdgeInsets.all(8),
      child: pw.Text(text, style: pw.TextStyle(fontSize: 9, fontWeight: bold ? pw.FontWeight.bold : pw.FontWeight.normal, color: _dark), textAlign: pw.TextAlign.center),
    );
  }

  static pw.Widget _buildModelCard(Map<String, dynamic> m, bool isBest) {
    return pw.Container(
      width: double.infinity,
      margin: const pw.EdgeInsets.only(bottom: 10),
      padding: const pw.EdgeInsets.all(14),
      decoration: pw.BoxDecoration(
        border: pw.Border.all(color: isBest ? _teal : PdfColors.grey300, width: isBest ? 1.5 : 0.5),
        borderRadius: pw.BorderRadius.circular(6),
      ),
      child: pw.Column(crossAxisAlignment: pw.CrossAxisAlignment.start, children: [
        pw.Row(children: [
          pw.Text(m['name'] as String, style: pw.TextStyle(fontSize: 13, fontWeight: pw.FontWeight.bold, color: _dark)),
          if (isBest) ...[
            pw.SizedBox(width: 8),
            pw.Container(
              padding: const pw.EdgeInsets.symmetric(horizontal: 6, vertical: 2),
              decoration: pw.BoxDecoration(color: PdfColor.fromInt(0xFFE0F2F1), borderRadius: pw.BorderRadius.circular(4)),
              child: pw.Text('BEST', style: pw.TextStyle(fontSize: 7, fontWeight: pw.FontWeight.bold, color: _teal)),
            ),
          ],
          pw.Spacer(),
          if (m['time'] != null)
            pw.Text('Training: ${(m['time'] as double).toStringAsFixed(1)}s', style: pw.TextStyle(fontSize: 8, color: _gray)),
        ]),
        pw.SizedBox(height: 10),
        pw.Row(mainAxisAlignment: pw.MainAxisAlignment.spaceBetween, children: [
          _metricBox('Accuracy', ((m['accuracy'] as double) * 100).toStringAsFixed(1)),
          _metricBox('Precision', ((m['precision'] as double) * 100).toStringAsFixed(1)),
          _metricBox('Recall', ((m['recall'] as double) * 100).toStringAsFixed(1)),
          _metricBox('F1 Score', ((m['f1'] as double) * 100).toStringAsFixed(1)),
          _metricBox('AUC', (m['auc'] as double).toStringAsFixed(3)),
        ]),
      ]),
    );
  }

  static pw.Widget _metricBox(String label, String value) {
    return pw.Column(children: [
      pw.Text(value, style: pw.TextStyle(fontSize: 11, fontWeight: pw.FontWeight.bold, color: _teal)),
      pw.SizedBox(height: 2),
      pw.Text(label, style: pw.TextStyle(fontSize: 7, color: _gray)),
    ]);
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  Save & Open
  // ══════════════════════════════════════════════════════════════════════════

  static Future<void> _saveAndOpen(pw.Document pdf, String fileName) async {
    final dir = await getApplicationDocumentsDirectory();
    final timestamp = DateTime.now().millisecondsSinceEpoch;
    final file = File('${dir.path}/${fileName}_$timestamp.pdf');
    await file.writeAsBytes(await pdf.save());
    await OpenFilex.open(file.path);
  }
}
