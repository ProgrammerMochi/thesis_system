package ph.thesis.smishing

import android.app.Activity
import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import android.text.SpannableString
import android.text.Spanned
import android.text.style.BackgroundColorSpan
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.ProgressBar
import android.widget.TextView
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.math.abs

/**
 * One screen: paste, share or select a message, mask it on the phone, send it
 * to the server, then show the verdict and the word highlights.
 *
 * Two requests per check: explain=false first, so the verdict appears after
 * classification alone, then explain=true for the SHAP highlights, which take
 * seconds on a laptop CPU. The timing line reports both separately.
 */
class MainActivity : Activity() {
    private val worker = Executors.newSingleThreadExecutor()
    // Results of an older check are dropped if the user started a new one.
    @Volatile private var generation = 0

    private lateinit var serverUrl: EditText
    private lateinit var message: EditText
    private lateinit var check: Button
    private lateinit var progress: ProgressBar
    private lateinit var status: TextView
    private lateinit var verdict: TextView
    private lateinit var probability: TextView
    private lateinit var highlighted: TextView
    private lateinit var legend: TextView
    private lateinit var urlFlags: TextView
    private lateinit var sentText: TextView
    private lateinit var timing: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        serverUrl = findViewById(R.id.serverUrl)
        message = findViewById(R.id.message)
        check = findViewById(R.id.check)
        progress = findViewById(R.id.progress)
        status = findViewById(R.id.status)
        verdict = findViewById(R.id.verdict)
        probability = findViewById(R.id.probability)
        highlighted = findViewById(R.id.highlighted)
        legend = findViewById(R.id.legend)
        urlFlags = findViewById(R.id.urlFlags)
        sentText = findViewById(R.id.sentText)
        timing = findViewById(R.id.timing)

        // Only the server address is remembered, never a message.
        serverUrl.setText(getPreferences(MODE_PRIVATE).getString(PREF_SERVER, ""))
        check.setOnClickListener { runCheck() }
        findViewById<Button>(R.id.clear).setOnClickListener {
            generation++
            message.setText("")
            clearResult()
            setBusy(false)
        }
        if (savedInstanceState == null) handleIntent(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleIntent(intent)
    }

    override fun onDestroy() {
        worker.shutdownNow()
        super.onDestroy()
    }

    /** Share sheet (ACTION_SEND) and text selection (ACTION_PROCESS_TEXT). */
    private fun handleIntent(intent: Intent) {
        val shared = when (intent.action) {
            Intent.ACTION_SEND -> intent.getStringExtra(Intent.EXTRA_TEXT)
            Intent.ACTION_PROCESS_TEXT -> intent.getCharSequenceExtra(Intent.EXTRA_PROCESS_TEXT)?.toString()
            else -> null
        }
        if (!shared.isNullOrBlank()) {
            message.setText(shared)
            if (serverUrl.text.isNotBlank()) runCheck()
        }
    }

    private fun runCheck() {
        val original = message.text.toString()
        val base = serverUrl.text.toString().trim()
        clearResult()
        if (original.isBlank()) { status.text = "Enter a message first."; return }
        if (!base.startsWith("http://") && !base.startsWith("https://")) {
            status.text = "Enter the server address, starting with http:// (laptop) or https://."
            return
        }
        // The server limit (SMISHING_MAX_CHARS, 1600) counts characters.
        if (original.codePointCount(0, original.length) > MAX_CHARS) {
            status.text = "Message is longer than $MAX_CHARS characters."
            return
        }
        getPreferences(MODE_PRIVATE).edit().putString(PREF_SERVER, base).apply()

        val masked = Masker.mask(original)
        sentText.text = "Sent to server (masked on this phone): ${masked.text}"
        val gen = ++generation
        val client = AnalyzeClient(base)
        setBusy(true)
        status.text = "Checking…"

        worker.execute {
            try {
                val fast = client.analyze(masked.text, explain = false)
                ui(gen) {
                    showVerdict(fast)
                    status.text = "Computing word highlights…"
                }
                val full = client.analyze(masked.text, explain = true)
                ui(gen) {
                    showVerdict(full)
                    showHighlights(original, masked, full)
                    timing.text = timingLine(fast, full)
                    status.text = if (full.explanationComputed) "" else "Highlights not available for this message."
                    setBusy(false)
                }
            } catch (e: Exception) {
                // Exception messages from the network stack never contain message text.
                ui(gen) {
                    status.text = "Could not check the message: ${e.javaClass.simpleName}" +
                        (e.message?.let { ": $it" } ?: "")
                    setBusy(false)
                }
            }
        }
    }

    private fun ui(gen: Int, block: () -> Unit) = runOnUiThread {
        if (gen == generation && !isDestroyed) block()
    }

    private fun showVerdict(r: AnalyzeResult) {
        val malicious = r.verdict == "malicious"
        verdict.text = if (malicious) "Likely MALICIOUS (smishing or spam)" else "Likely LEGITIMATE"
        verdict.setTextColor(if (malicious) RED else GREEN)
        probability.text = String.format(
            Locale.US, "Threat probability %.1f%% (threshold %.0f%%; %s)",
            r.threatProbability * 100, r.threshold * 100, r.basis,
        )
        urlFlags.text = if (r.urlFlags.isEmpty()) "Link warnings: none"
        else "Link warnings: " + r.urlFlags.joinToString(", ") { FLAG_TEXT[it] ?: it }
    }

    /**
     * Server offsets are UTF-16 units into the masked text; Masked maps them
     * back onto the original message. Colour strength is |weight| relative to
     * the largest |weight| in this message, so it compares words within one
     * message only, not across messages.
     */
    private fun showHighlights(original: String, masked: Masked, r: AnalyzeResult) {
        val spans = SpannableString(original)
        val maxAbs = r.attributions.maxOfOrNull { abs(it.weight) } ?: 0.0
        if (maxAbs > 0.0) {
            for (a in r.attributions) {
                val (s, e) = masked.toOriginal(a.start, a.end) ?: continue
                val alpha = (30 + 170 * abs(a.weight) / maxAbs).toInt().coerceIn(0, 255)
                val rgb = if (a.weight > 0) RED else GREEN
                val color = Color.argb(alpha, Color.red(rgb), Color.green(rgb), Color.blue(rgb))
                spans.setSpan(BackgroundColorSpan(color), s, e, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
            }
        }
        highlighted.text = spans
        legend.visibility = if (r.attributions.isEmpty()) View.GONE else View.VISIBLE
        if (r.truncated) legend.append("\nThe message was longer than the model reads; the end was not explained.")
    }

    private fun timingLine(fast: AnalyzeResult, full: AnalyzeResult): String {
        fun ms(r: AnalyzeResult, k: String) = r.serverTiming[k]?.let { String.format(Locale.US, "%.0f ms", it) } ?: "n/a"
        return "Verdict: ${fast.roundTripMs} ms round trip (classify ${ms(fast, "classify")}). " +
            "Highlights: ${full.roundTripMs} ms round trip (SHAP ${ms(full, "shap")}). " +
            "Model ${full.modelVersion}."
    }

    private fun clearResult() {
        listOf(status, verdict, probability, highlighted, urlFlags, sentText, timing).forEach { it.text = "" }
        legend.text = getString(R.string.legend)
        legend.visibility = View.GONE
    }

    private fun setBusy(busy: Boolean) {
        check.isEnabled = !busy
        progress.visibility = if (busy) View.VISIBLE else View.GONE
    }

    companion object {
        private const val PREF_SERVER = "server_url"
        private const val MAX_CHARS = 1600
        private val RED = Color.rgb(0xC6, 0x28, 0x28)
        private val GREEN = Color.rgb(0x2E, 0x7D, 0x32)
        // Plain-language names for ml/url_features.py FLAG_NAMES.
        private val FLAG_TEXT = mapOf(
            "ip_host" to "link uses a bare IP address",
            "shortener_domain" to "link shortener",
            "punycode_host" to "look-alike (punycode) domain",
            "at_in_url" to "'@' inside the link",
            "no_https" to "link does not use https",
            "nonstandard_port" to "unusual port in the link",
            "malformed_url" to "malformed link",
        )
    }
}
