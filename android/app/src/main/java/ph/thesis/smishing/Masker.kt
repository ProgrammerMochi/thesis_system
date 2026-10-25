package ph.thesis.smishing

import java.util.regex.Matcher
import java.util.regex.Pattern

/**
 * On-device masking of phone numbers and OTPs (Chapter 3, Section 3.5.2),
 * before anything leaves the phone.
 *
 * The two patterns are ports of PHONE_RE and OTP_RE in ml/textnorm.py and
 * must stay identical to them; tests/test_textnorm.py is the contract, and
 * its PHONE and OTP cases are repeated in MaskerTest. Everything else
 * (URLs, emails, amounts, reference numbers) is masked by the server, which
 * runs ml/textnorm.py on what this class sends.
 *
 * Python 3 regexes are Unicode-aware (\d, \s, \b and case-insensitivity),
 * Java's are ASCII by default, so UNICODE_CHARACTER_CLASS is requested. If a
 * platform rejects that flag the patterns fall back to ASCII classes and
 * [unicodeClasses] is false; the only inputs that then differ are non-ASCII
 * digits, non-ASCII spaces (e.g. a no-break space between "OTP" and the
 * code) and non-ASCII letters touching a keyword.
 *
 * Besides the masked text, the class keeps, for every masked character, the
 * span of original characters that produced it (the same bookkeeping as
 * _tsub in ml/textnorm.py), so highlights the server returns for the masked
 * text can be drawn on the message the user actually sees. Offsets are
 * Kotlin String indices, i.e. UTF-16 units, the unit the server reports.
 */
class Masked internal constructor(
    val text: String,
    private val starts: IntArray,
    private val ends: IntArray,
) {
    /** Maps [start, end) of the masked text to the original text, or null if out of range. */
    fun toOriginal(start: Int, end: Int): Pair<Int, Int>? {
        if (start < 0 || end > text.length || start >= end) return null
        var s = Int.MAX_VALUE
        var e = Int.MIN_VALUE
        for (i in start until end) {
            if (starts[i] < s) s = starts[i]
            if (ends[i] > e) e = ends[i]
        }
        return s to e
    }
}

object Masker {
    private const val PHONE =
        """(?<!\d)(?:\+?63|0)9\d{2}[\s-]?\d{3}[\s-]?\d{4}(?!\d)"""
    private const val OTP =
        """\b(otp|one[- ]time (?:pin|password)|verification code|code|pin)(\s*(?:is|:|ay)?\s*)(\d{4,8})\b"""

    var unicodeClasses = true
        private set

    private val phoneRe: Pattern = compile(PHONE, 0)
    private val otpRe: Pattern = compile(OTP, Pattern.CASE_INSENSITIVE or Pattern.UNICODE_CASE)

    private fun compile(regex: String, flags: Int): Pattern =
        try {
            Pattern.compile(regex, flags or Pattern.UNICODE_CHARACTER_CLASS)
        } catch (e: IllegalArgumentException) {
            unicodeClasses = false
            Pattern.compile(regex, flags)
        }

    /** A piece of output text and the [start, end) of the input it stands for. */
    private class Piece(val text: String, val start: Int, val end: Int)

    fun mask(original: String): Masked {
        var t = original
        var starts = IntArray(t.length) { it }
        var ends = IntArray(t.length) { it + 1 }

        fun replaceTracked(re: Pattern, pieces: (Matcher) -> List<Piece>) {
            val out = StringBuilder()
            val outS = ArrayList<Int>()
            val outE = ArrayList<Int>()
            fun copy(a: Int, b: Int) {
                out.append(t, a, b)
                for (i in a until b) { outS.add(starts[i]); outE.add(ends[i]) }
            }
            val m = re.matcher(t)
            var pos = 0
            while (m.find()) {
                copy(pos, m.start())
                for (p in pieces(m)) {
                    if (t.substring(p.start, p.end) == p.text) {
                        copy(p.start, p.end)
                    } else {
                        // A replacement token covers the whole matched input.
                        var s = Int.MAX_VALUE
                        var e = Int.MIN_VALUE
                        for (i in p.start until p.end) {
                            if (starts[i] < s) s = starts[i]
                            if (ends[i] > e) e = ends[i]
                        }
                        if (p.start == p.end) { s = 0; e = 0 }
                        out.append(p.text)
                        repeat(p.text.length) { outS.add(s); outE.add(e) }
                    }
                }
                pos = m.end()
            }
            copy(pos, t.length)
            t = out.toString()
            starts = outS.toIntArray()
            ends = outE.toIntArray()
        }

        // Same order as mask_pii(): phone numbers, then OTPs.
        replaceTracked(phoneRe) { m -> listOf(Piece("[PHONE_NUM]", m.start(), m.end())) }
        replaceTracked(otpRe) { m ->
            listOf(
                Piece(m.group(1)!!, m.start(1), m.end(1)),
                Piece(m.group(2)!!, m.start(2), m.end(2)),
                Piece("[OTP]", m.start(3), m.end(3)),
            )
        }
        return Masked(t, starts, ends)
    }
}
