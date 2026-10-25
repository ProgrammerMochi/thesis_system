package ph.thesis.smishing

import org.json.JSONObject
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

data class Attribution(val start: Int, val end: Int, val weight: Double)

/** The Figure 3 response of POST /v1/analyze (server/app/schemas.py). */
data class AnalyzeResult(
    val verdict: String,
    val threatProbability: Double,
    val attributions: List<Attribution>,
    val urlFlags: List<String>,
    val threshold: Double,
    val basis: String,
    val explanationComputed: Boolean,
    val truncated: Boolean,
    val modelVersion: String,
    /** Server-Timing header entries, name to milliseconds. */
    val serverTiming: Map<String, Double>,
    val roundTripMs: Long,
)

class ServerError(message: String) : IOException(message)

/**
 * Minimal client for the FastAPI server. Only the masked text is sent; the
 * server rejects any other field (data minimization). Nothing is logged or
 * stored on the phone: errors carry status codes and error types only, never
 * message text.
 */
class AnalyzeClient(baseUrl: String) {
    private val base = baseUrl.trim().trimEnd('/')

    fun analyze(maskedText: String, explain: Boolean): AnalyzeResult {
        val started = System.nanoTime()
        val conn = URL("$base/v1/analyze?explain=$explain").openConnection() as HttpURLConnection
        try {
            conn.requestMethod = "POST"
            conn.connectTimeout = 5_000
            // SHAP on a laptop CPU can take several seconds on long messages.
            conn.readTimeout = if (explain) 90_000 else 15_000
            conn.doOutput = true
            conn.useCaches = false
            conn.setRequestProperty("Content-Type", "application/json; charset=utf-8")
            conn.setRequestProperty("Accept", "application/json")
            val body = JSONObject().put("text", maskedText).toString().toByteArray(Charsets.UTF_8)
            conn.setFixedLengthStreamingMode(body.size)
            conn.outputStream.use { it.write(body) }

            val code = conn.responseCode
            val stream = if (code in 200..299) conn.inputStream else conn.errorStream
            val raw = stream?.bufferedReader(Charsets.UTF_8)?.use { it.readText() } ?: ""
            if (code !in 200..299) {
                val type = runCatching { JSONObject(raw).optString("error") }.getOrNull()
                throw ServerError("server answered $code${if (type.isNullOrEmpty()) "" else " ($type)"}")
            }
            val elapsed = (System.nanoTime() - started) / 1_000_000
            return parse(JSONObject(raw), conn.getHeaderField("Server-Timing"), elapsed)
        } finally {
            conn.disconnect()
        }
    }

    private fun parse(j: JSONObject, timing: String?, roundTripMs: Long): AnalyzeResult {
        val attrs = j.getJSONArray("text_attributions")
        val flags = j.getJSONArray("url_flags")
        val decision = j.getJSONObject("decision")
        val explanation = j.getJSONObject("explanation")
        return AnalyzeResult(
            verdict = j.getString("verdict"),
            threatProbability = j.getDouble("threat_probability"),
            attributions = (0 until attrs.length()).map {
                val a = attrs.getJSONObject(it)
                Attribution(a.getInt("start"), a.getInt("end"), a.getDouble("weight"))
            },
            urlFlags = (0 until flags.length()).map { flags.getString(it) },
            threshold = decision.getDouble("threshold"),
            basis = decision.getString("basis"),
            explanationComputed = explanation.getBoolean("computed"),
            truncated = explanation.getBoolean("truncated"),
            modelVersion = j.getString("model_version"),
            serverTiming = parseServerTiming(timing),
            roundTripMs = roundTripMs,
        )
    }

    companion object {
        /** "total;dur=812.3, shap;dur=640.1" -> {total=812.3, shap=640.1}. */
        fun parseServerTiming(header: String?): Map<String, Double> {
            if (header.isNullOrBlank()) return emptyMap()
            val out = LinkedHashMap<String, Double>()
            for (entry in header.split(',')) {
                val parts = entry.trim().split(';')
                val name = parts[0].trim()
                val dur = parts.drop(1).map { it.trim() }
                    .firstOrNull { it.startsWith("dur=") }?.removePrefix("dur=")?.toDoubleOrNull()
                if (name.isNotEmpty() && dur != null) out[name] = dur
            }
            return out
        }
    }
}
