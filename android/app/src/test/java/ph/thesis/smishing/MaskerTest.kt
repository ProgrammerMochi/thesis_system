package ph.thesis.smishing

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Contract with ml/textnorm.py. The expected strings were produced by
 * PHONE_RE.sub then OTP_RE.sub in Python (the phone/OTP part of mask_pii);
 * the first three inputs are the PHONE and OTP cases of tests/test_textnorm.py.
 * If either side changes, regenerate these from Python, never by hand.
 *
 * Runs on the desktop JVM. Android's regex engine is ICU-based, so also run
 * the app on a device with the no-break-space case before the demo.
 */
class MaskerTest {
    private val cases = listOf(
        "Get P3K now! visit lvv65a118l.uk/?x=1. Call 0917-123-4567" to
            "Get P3K now! visit lvv65a118l.uk/?x=1. Call [PHONE_NUM]",
        "Your OTP is 482910. Ref 1234567890" to "Your OTP is [OTP]. Ref 1234567890",
        "Login sa GoJackpot1.vegas, +639171234567" to "Login sa GoJackpot1.vegas, [PHONE_NUM]",
        "639171234567" to "[PHONE_NUM]",
        "0917 123 4567" to "[PHONE_NUM]",
        "109171234567" to "109171234567",
        "09171234567890" to "09171234567890",
        "Code: 1234" to "Code: [OTP]",
        "verification code ay 123456" to "verification code ay [OTP]",
        "PIN 123" to "PIN 123",
        "Your one-time password is 55667788" to "Your one-time password is [OTP]",
        "OTP 123456" to "OTP [OTP]",
        "😀 Code 1234" to "😀 Code [OTP]",
        "pin12345" to "pin[OTP]",
    )

    @Test fun matchesPythonMasking() {
        assertTrue("UNICODE_CHARACTER_CLASS rejected on this JVM", Masker.unicodeClasses)
        for ((raw, want) in cases) assertEquals(raw, want, Masker.mask(raw).text)
    }

    @Test fun phoneOffsetsMapBack() {
        val m = Masker.mask("Call 0917-123-4567 now")
        assertEquals("Call [PHONE_NUM] now", m.text)
        assertEquals(5 to 18, m.toOriginal(5, 16))   // the token covers the whole number
        assertEquals(19 to 22, m.toOriginal(17, 20)) // "now" shifts back by 2
        assertEquals(0 to 4, m.toOriginal(0, 4))
    }

    @Test fun otpOffsetsMapBack() {
        val m = Masker.mask("Your OTP is 482910.")
        assertEquals(12 to 18, m.toOriginal(12, 17))
        assertEquals(18 to 19, m.toOriginal(17, 18))
        assertEquals(5 to 8, m.toOriginal(5, 8))     // the keyword itself is kept
    }

    @Test fun emojiIsTwoUtf16Units() {
        val m = Masker.mask("😀 Code 1234")
        assertEquals(0 to 2, m.toOriginal(0, 2))
        assertEquals(3 to 7, m.toOriginal(3, 7))
        assertEquals(8 to 12, m.toOriginal(8, 13))
    }

    @Test fun outOfRangeIsNull() {
        val m = Masker.mask("hello")
        assertNull(m.toOriginal(3, 9))
        assertNull(m.toOriginal(2, 2))
        assertNull(m.toOriginal(-1, 2))
    }

    @Test fun serverTimingHeader() {
        val t = AnalyzeClient.parseServerTiming("queue;dur=0.1, classify;dur=85.2, shap;dur=2830.0, total;dur=2950.4")
        assertEquals(85.2, t["classify"]!!, 1e-9)
        assertEquals(2830.0, t["shap"]!!, 1e-9)
        assertEquals(emptyMap<String, Double>(), AnalyzeClient.parseServerTiming(null))
    }
}
