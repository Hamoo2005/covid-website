import React, { useState } from "react";
import { jsPDF } from "jspdf";


const API_BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:10000";

const personalInitialState = {
  patientName: "",
  patientAddress: "",
  patientPhone: "",
};

const medicalInitialState = {
  MEDICAL_UNIT: "",
  PATIENT_TYPE: "",
  INTUBED: "",
  PNEUMONIA: "",
  AGE: "",
  DIABETES: "",
  HIPERTENSION: "",
  RENAL_CHRONIC: "",
  CLASIFFICATION_FINAL: "",
  ICU: "",
};

const patientTypeOptions = [
  { value: "1", label: "1 - Outpatient" },
  { value: "2", label: "2 - Inpatient" },
];

const yesNoOptions = [
  { value: "Yes", label: "Yes" },
  { value: "No", label: "No" },
];

const classificationOptions = [
  { value: "1", label: "1 - COVID Positive" },
  { value: "2", label: "2 - COVID Positive" },
  { value: "3", label: "3 - COVID Positive" },
  { value: "4", label: "4 - Not COVID" },
  { value: "5", label: "5 - Not COVID" },
  { value: "6", label: "6 - Not COVID" },
  { value: "7", label: "7 - Not COVID" },
];

const medicalFieldLabels = {
  MEDICAL_UNIT: "Medical Unit",
  PATIENT_TYPE: "Patient Type",
  INTUBED: "Intubed",
  PNEUMONIA: "Pneumonia",
  AGE: "Age",
  DIABETES: "Diabetes",
  HIPERTENSION: "Hypertension",
  RENAL_CHRONIC: "Renal Chronic",
  CLASIFFICATION_FINAL: "COVID Classification",
  ICU: "ICU",
};

const numericFields = new Set([
  "MEDICAL_UNIT",
  "PATIENT_TYPE",
  "AGE",
  "CLASIFFICATION_FINAL",
]);

function App() {
  const [personalInfo, setPersonalInfo] = useState(personalInitialState);
  const [medicalInfo, setMedicalInfo] = useState(medicalInitialState);
  const [result, setResult] = useState(null);
  const [reportDateTime, setReportDateTime] = useState("");
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const displayValues = {
    PATIENT_TYPE:
      patientTypeOptions.find((option) => option.value === medicalInfo.PATIENT_TYPE)
        ?.label || medicalInfo.PATIENT_TYPE,
    INTUBED: medicalInfo.INTUBED,
    PNEUMONIA: medicalInfo.PNEUMONIA,
    DIABETES: medicalInfo.DIABETES,
    HIPERTENSION: medicalInfo.HIPERTENSION,
    RENAL_CHRONIC: medicalInfo.RENAL_CHRONIC,
    ICU: medicalInfo.ICU,
    CLASIFFICATION_FINAL:
      classificationOptions.find(
        (option) => option.value === medicalInfo.CLASIFFICATION_FINAL
      )?.label || medicalInfo.CLASIFFICATION_FINAL,
  };

  const updatePersonalField = (event) => {
    const { name, value } = event.target;
    setPersonalInfo((current) => ({ ...current, [name]: value }));
  };

  const updateMedicalField = (event) => {
    const { name, value } = event.target;
    setMedicalInfo((current) => ({ ...current, [name]: value }));
  };

  const validateForm = () => {
    if (!personalInfo.patientName.trim()) {
      return "Patient Name is required.";
    }
    if (!personalInfo.patientAddress.trim()) {
      return "Patient Address is required.";
    }
    if (!personalInfo.patientPhone.trim()) {
      return "Patient Phone Number is required.";
    }

    for (const [field, value] of Object.entries(medicalInfo)) {
      if (!String(value).trim()) {
        return `${medicalFieldLabels[field]} is required.`;
      }
    }

    const age = Number(medicalInfo.AGE);
    if (Number.isNaN(age) || age < 0 || age > 120) {
      return "Age must be between 0 and 120.";
    }

    return "";
  };

  const buildMedicalPayload = () => {
    const normalized = {};

    for (const [field, value] of Object.entries(medicalInfo)) {
      if (numericFields.has(field)) {
        normalized[field] = Number(value);
      } else {
        normalized[field] = String(value).trim();
      }
    }

    return normalized;
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError("");

    const validationMessage = validateForm();
    if (validationMessage) {
      setResult(null);
      setError(validationMessage);
      return;
    }

    setIsSubmitting(true);

    try {
      const medicalPayload = buildMedicalPayload();
      console.log("Submitting medical payload:", medicalPayload);

      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          patient_info: personalInfo,
          medical_inputs: medicalPayload,
        }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Prediction failed.");
      }

      setResult(data);
      setReportDateTime(new Date().toLocaleString());
    } catch (requestError) {
      setResult(null);
      setError(
        requestError.message ||
          "Unable to connect to the prediction service. Please try again."
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  const resetForm = () => {
    setPersonalInfo(personalInitialState);
    setMedicalInfo(medicalInitialState);
    setResult(null);
    setReportDateTime("");
    setError("");
  };

  const downloadPdfReport = () => {
    if (!result) {
      return;
    }

    const doc = new jsPDF();
    const createdAt = reportDateTime || new Date().toLocaleString();
    let y = 20;

    const addLine = (text, value, isMultiline = false) => {
      doc.setFont("helvetica", "bold");
      doc.text(text, 16, y);
      doc.setFont("helvetica", "normal");

      if (isMultiline) {
        const wrapped = doc.splitTextToSize(value, 120);
        doc.text(wrapped, 78, y);
        y += wrapped.length * 7;
      } else {
        doc.text(String(value), 78, y);
        y += 8;
      }
    };

    doc.setFillColor(13, 86, 153);
    doc.roundedRect(12, 10, 186, 24, 6, 6, "F");
    doc.setTextColor(255, 255, 255);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(18);
    doc.text("COVID-19 Patient Risk Prediction Report", 16, 25);

    doc.setTextColor(17, 49, 77);
    doc.setFontSize(11);
    y = 46;

    doc.setFont("helvetica", "bold");
    doc.text("Report Details", 16, y);
    y += 8;
    addLine("Date and Time", createdAt);

    y += 4;
    doc.setFont("helvetica", "bold");
    doc.text("Patient Information", 16, y);
    y += 8;
    addLine("Patient Name", personalInfo.patientName);
    addLine("Patient Address", personalInfo.patientAddress, true);
    addLine("Patient Phone Number", personalInfo.patientPhone);

    y += 4;
    doc.setFont("helvetica", "bold");
    doc.text("Medical Inputs", 16, y);
    y += 8;
    addLine("Medical Unit", medicalInfo.MEDICAL_UNIT);
    addLine("Patient Type", displayValues.PATIENT_TYPE);
    addLine("Intubed", displayValues.INTUBED);
    addLine("Pneumonia", displayValues.PNEUMONIA);
    addLine("Age", medicalInfo.AGE);
    addLine("Diabetes", displayValues.DIABETES);
    addLine("Hypertension", displayValues.HIPERTENSION);
    addLine("Renal Chronic", displayValues.RENAL_CHRONIC);
    addLine("COVID Classification", displayValues.CLASIFFICATION_FINAL);
    addLine("ICU", displayValues.ICU);

    y += 4;
    doc.setFillColor(238, 246, 255);
    doc.roundedRect(14, y, 180, 32, 4, 4, "F");
    y += 10;
    doc.setFont("helvetica", "bold");
    doc.text("Prediction Result", 18, y);
    y += 8;
    doc.setFont("helvetica", "normal");
    doc.text(`Result: ${result.result}`, 18, y);
    y += 8;
    doc.text(`Survived Probability: ${result.survived_probability}%`, 18, y);
    y += 8;
    doc.text(`Death Probability: ${result.death_probability}%`, 18, y);

    doc.setFontSize(10);
    doc.setTextColor(93, 120, 146);
    doc.text("By Hamo", 105, 286, { align: "center" });

    const safeName =
      personalInfo.patientName.trim().replace(/[^a-z0-9]+/gi, "_").toLowerCase() ||
      "patient";
    doc.save(`covid-risk-report-${safeName}.pdf`);
  };

  return (
    <div className="app-shell">
      <header className="hero-card">
        <div>
          <span className="hero-badge">Medical Dashboard</span>
          <h1>COVID-19 Mortality Risk Prediction</h1>
          <p>
            Review patient information, submit the clinical indicators, and generate a
            professional PDF report after prediction.
          </p>
        </div>

        <div className="hero-side-card">
          <h2>Model Workflow</h2>
          <ul>
            <li>Saved Flask model loaded from `.pkl` files</li>
            <li>Only medical inputs are sent into the ML pipeline</li>
            <li>Patient personal details are used only for reporting</li>
          </ul>
        </div>
      </header>

      <main className="dashboard-grid">
        <section className="main-panel">
          <form className="dashboard-card form-card" onSubmit={handleSubmit}>
            <div className="section-heading">
              <div>
                <span className="section-label">Patient Record</span>
                <h2>Personal Information</h2>
              </div>
            </div>

            <div className="form-grid">
              <label className="field">
                <span>Patient Name</span>
                <input
                  type="text"
                  name="patientName"
                  placeholder="Enter full name"
                  value={personalInfo.patientName}
                  onChange={updatePersonalField}
                />
              </label>

              <label className="field">
                <span>Patient Phone Number</span>
                <input
                  type="tel"
                  name="patientPhone"
                  placeholder="Enter phone number"
                  value={personalInfo.patientPhone}
                  onChange={updatePersonalField}
                />
              </label>

              <label className="field field-full">
                <span>Patient Address</span>
                <input
                  type="text"
                  name="patientAddress"
                  placeholder="Enter home address"
                  value={personalInfo.patientAddress}
                  onChange={updatePersonalField}
                />
              </label>
            </div>

            <div className="section-heading section-separator">
              <div>
                <span className="section-label">Clinical Inputs</span>
                <h2>Medical Information</h2>
              </div>
            </div>

            <div className="form-grid">
              <label className="field">
                <span>Medical Unit</span>
                <input
                  type="number"
                  name="MEDICAL_UNIT"
                  placeholder="Example: 1 or 2 or 3"
                  min="1"
                  value={medicalInfo.MEDICAL_UNIT}
                  onChange={updateMedicalField}
                />
              </label>

              <label className="field">
                <span>Patient Type</span>
                <select
                  name="PATIENT_TYPE"
                  value={medicalInfo.PATIENT_TYPE}
                  onChange={updateMedicalField}
                >
                  <option value="">Select patient type</option>
                  {patientTypeOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>

              <label className="field">
                <span>Intubed</span>
                <select name="INTUBED" value={medicalInfo.INTUBED} onChange={updateMedicalField}>
                  <option value="">Select an option</option>
                  {yesNoOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
                <small>Yes = patient needs breathing machine</small>
              </label>

              <label className="field">
                <span>Pneumonia</span>
                <select
                  name="PNEUMONIA"
                  value={medicalInfo.PNEUMONIA}
                  onChange={updateMedicalField}
                >
                  <option value="">Select an option</option>
                  {yesNoOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>

              <label className="field">
                <span>Age</span>
                <input
                  type="number"
                  name="AGE"
                  placeholder="Example: 25, 40, 70"
                  min="0"
                  max="120"
                  value={medicalInfo.AGE}
                  onChange={updateMedicalField}
                />
              </label>

              <label className="field">
                <span>Diabetes</span>
                <select
                  name="DIABETES"
                  value={medicalInfo.DIABETES}
                  onChange={updateMedicalField}
                >
                  <option value="">Select an option</option>
                  {yesNoOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>

              <label className="field">
                <span>Hypertension</span>
                <select
                  name="HIPERTENSION"
                  value={medicalInfo.HIPERTENSION}
                  onChange={updateMedicalField}
                >
                  <option value="">Select an option</option>
                  {yesNoOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>

              <label className="field">
                <span>Renal Chronic</span>
                <select
                  name="RENAL_CHRONIC"
                  value={medicalInfo.RENAL_CHRONIC}
                  onChange={updateMedicalField}
                >
                  <option value="">Select an option</option>
                  {yesNoOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>

              <label className="field">
                <span>COVID Classification</span>
                <select
                  name="CLASIFFICATION_FINAL"
                  value={medicalInfo.CLASIFFICATION_FINAL}
                  onChange={updateMedicalField}
                >
                  <option value="">Select classification</option>
                  {classificationOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>

              <label className="field">
                <span>ICU</span>
                <select name="ICU" value={medicalInfo.ICU} onChange={updateMedicalField}>
                  <option value="">Select an option</option>
                  {yesNoOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
                <small>Yes = patient entered intensive care</small>
              </label>
            </div>

            {error ? <div className="alert-error">{error}</div> : null}

            <div className="action-row">
              <button className="primary-button" type="submit" disabled={isSubmitting}>
                {isSubmitting ? "Predicting..." : "Predict Risk"}
              </button>
              <button className="secondary-button" type="button" onClick={resetForm}>
                New Patient
              </button>
            </div>
          </form>
        </section>

        <aside className="side-panel">
          <div className="dashboard-card result-card">
            <div className="section-heading">
              <div>
                <span className="section-label">Prediction Output</span>
                <h2>Clinical Summary</h2>
              </div>
            </div>

            {result ? (
              <>
                <div
                  className={`risk-banner ${
                    result.result === "High Risk (Death)" ? "high-risk" : "low-risk"
                  }`}
                >
                  {result.result}
                </div>

                <div className="metric-card">
                  <div className="metric-row">
                    <span>Survived Probability</span>
                    <strong>{result.survived_probability}%</strong>
                  </div>
                  <div className="progress-track">
                    <div
                      className="progress-fill survived"
                      style={{ width: `${result.survived_probability}%` }}
                    />
                  </div>
                </div>

                <div className="metric-card">
                  <div className="metric-row">
                    <span>Death Probability</span>
                    <strong>{result.death_probability}%</strong>
                  </div>
                  <div className="progress-track">
                    <div
                      className="progress-fill death"
                      style={{ width: `${result.death_probability}%` }}
                    />
                  </div>
                </div>

                <div className="report-meta">
                  <span>Report timestamp</span>
                  <strong>{reportDateTime}</strong>
                </div>

                <button className="primary-button full-width" onClick={downloadPdfReport}>
                  Print / Download PDF Report
                </button>

                {result.debug ? (
                  <details className="debug-panel">
                    <summary>Temporary Debug Data</summary>
                    <pre>{JSON.stringify(result.debug, null, 2)}</pre>
                  </details>
                ) : null}
              </>
            ) : (
              <div className="empty-state">
                <h3>No prediction yet</h3>
                <p>
                  Complete the patient form and submit it to see the prediction result and
                  export the PDF report.
                </p>
              </div>
            )}
          </div>

          <div className="dashboard-card info-card">
            <h3>Report Notes</h3>
            <ul>
              <li>Patient name, address, and phone are for the report only.</li>
              <li>The model uses only the medical inputs listed in the form.</li>
              <li>The PDF includes the full patient summary and prediction outcome.</li>
            </ul>
          </div>
        </aside>
      </main>

      <footer className="site-footer">By Hamo</footer>
    </div>
  );
}

export default App;
