plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "ph.thesis.smishing"
    compileSdk = 34

    defaultConfig {
        applicationId = "ph.thesis.smishing"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "0.1-defense-prototype"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

// No third-party runtime libraries: HTTP is HttpURLConnection and JSON is
// the platform's org.json, so the only moving parts are AGP and Kotlin.
dependencies {
    testImplementation("junit:junit:4.13.2")
}
