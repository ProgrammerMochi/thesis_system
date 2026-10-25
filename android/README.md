# Android client: defense prototype (Increment 4, minimal)

One screen: paste a message (or Share it / select it and pick the app), and the
app masks phone numbers and OTPs **on the phone**, sends the masked text to the
laptop server, and shows the verdict and the word highlights.

**Status: written but not yet built.** The machine it was written on has no
Android SDK. The first person to open it in Android Studio should build it,
run the unit tests, and report any errors. The masking regexes were checked on
the desktop Java engine against Python's `ml/textnorm.py` (14 of 14 cases
match); they still need checking on a real phone.

## What it does and does not do

| In this prototype | Not yet (later in Increment 4) |
| --- | --- |
| Paste, share sheet, text selection | Reading incoming SMS (`RECEIVE_SMS`) |
| On-device masking of phone numbers and OTPs, with an offset map so highlights land on the original text | Notifications, history, settings screen |
| Verdict first (`explain=false`), highlights second (`explain=true`) | URL fusion (Increment 2; the server decides on text only) |
| Plain HTTP in **debug builds only** (laptop demo) | HTTPS Cloud Run server (UAT) |

Nothing is logged or stored on the phone except the server address.

## Open and run (step by step)

1. Install Android Studio (one lead developer, on their own laptop).
2. In Android Studio: **File → Open** and pick the `android/` folder (not the repo root).
   If it offers to set up or upgrade the Gradle wrapper or the Android Gradle
   Plugin, accept. The project uses AGP 8.5.2, Kotlin 1.9.24 and Gradle 8.7.
3. Wait for the Gradle sync to finish, then run the unit tests:
   right-click `app/src/test/java/ph/thesis/smishing/MaskerTest` → **Run**.
   All tests must pass. If one fails, do not edit the expected strings by hand:
   tell a lead developer.
4. Start the server on the laptop (repo root, venv active). For building the
   app, the stub mode is enough and loads no model:

       SMISHING_MODE=stub python -m server

   For the real model: `python -m server`. If macOS asks whether Python may
   accept incoming connections, click **Allow**.
5. Find the laptop's address on the Wi-Fi or hotspot: `ipconfig getifaddr en0`.
6. Put the phone on the same Wi-Fi or hotspot, connect it with USB debugging
   on, and press **Run** in Android Studio.
7. In the app, enter `http://<laptop address>:8000` in the top box, paste a
   message and tap **Check message**.

If the app says it could not connect: check that both devices are on the same
network, the address and port are right, and the server terminal shows it is
running. Opening `http://<laptop address>:8000/healthz` in the phone's browser
should show `{"status":"ok"}`.

## Rules for changing it

- `Masker.kt` must stay identical in behaviour to `PHONE_RE` and `OTP_RE` in
  `ml/textnorm.py`. If those change, regenerate the cases in `MaskerTest`
  from Python.
- Never log or store message text (no `Log.d(text)`, no saving messages).
- Do not use test-split messages for demos or screenshots.

## Known gaps (state them if asked)

- The phone masks phone numbers and OTPs on the raw text; the server masks
  emails, links, amounts and reference numbers afterwards. In the rare case of
  a phone number inside an email address or link, the phone's masking changes
  what the server sees compared with training text.
- Highlight colour strength is relative within one message only.
- If the phone's regex engine rejects Unicode character classes, masking falls
  back to ASCII classes (only exotic digits and no-break spaces differ).
