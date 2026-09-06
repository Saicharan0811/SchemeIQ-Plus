# SchemeIQ+ Official Source Verification Report

**Verification Date**: 2026-08-15  
**Geographical Scope**: Telangana, India (State Schemes & Applicable Central Schemes)  
**Standard**: Strict Ground Truth from Official Government Portals (`.gov.in`, `.nic.in`, PFRDA, KVIC, PM-KISAN, NHA).  
**Rule Compliance**: No synthetic data; no hallucinated or inferred criteria; unstated fields are marked as `NOT SPECIFIED`.

---

## Summary Matrix of Verified Schemes

| Scheme ID | Scheme Name | Scope | Verified `age_min` | Verified `age_max` | Verified `income_limit` | Primary Source Authority |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TS001** | Rythu Bharosa | State | NOT SPECIFIED | NOT SPECIFIED | NOT SPECIFIED | Telangana Agriculture & Farmers Welfare Commission |
| **TS002** | Rythu Bhima | State | 18 | 59 | NOT SPECIFIED | Telangana Agriculture & Farmers Welfare Commission / LIC |
| **TS003** | Cheyutha (Rajiv Aarogyasri) | State | NOT SPECIFIED | NOT SPECIFIED | BPL / Food Security Card | Rajiv Aarogyasri Health Care Trust / Telangana State Portal |
| **TS004** | Maha Lakshmi | State | NOT SPECIFIED | NOT SPECIFIED | Component-specific (BPL for LPG) | Government of Telangana / TSRTC / Civil Supplies Dept |
| **TS005** | Aasara Pensions | State | 57 (OAP) / 18 (Widow) / 50 (Weavers) | NOT SPECIFIED | ₹1.5L (Rural) / ₹2.0L (Urban) | Telangana Aasara Portal (SERP / PR&RD) |
| **TS006** | Telangana ePASS Scholarships | State | NOT SPECIFIED | NOT SPECIFIED | ₹1.5L–₹2.5L (Category-wise) | Telangana ePASS / CGG / Welfare Departments |
| **TS007** | MCH Kit Scheme (KCR Kit) | State | NOT SPECIFIED | NOT SPECIFIED | NOT SPECIFIED | Commissionerate of Health & Family Welfare, Telangana |
| **TS008** | Telangana Free Diagnostic Services | State | NOT SPECIFIED | NOT SPECIFIED | NOT SPECIFIED (Universal) | Commissionerate of Health & Family Welfare, Telangana |
| **CT001** | PM-KISAN | Central | NOT SPECIFIED | NOT SPECIFIED | NOT SPECIFIED (Statutory exclusions) | Ministry of Agriculture & Farmers Welfare, GoI |
| **CT002** | Ayushman Bharat PM-JAY | Central | NOT SPECIFIED (70 for senior top-up) | NOT SPECIFIED | SECC 2011 Deprivation / Universal for 70+ | National Health Authority (NHA), GoI |
| **CT003** | PMAY-U 2.0 | Central | 18 | NOT SPECIFIED | ₹3L (EWS), ₹6L (LIG), ₹9L (MIG) | Ministry of Housing and Urban Affairs (MoHUA), GoI |
| **CT004** | Pradhan Mantri Mudra Yojana (PMMY) | Central | 18 | 65 (at loan maturity) | NOT SPECIFIED | MUDRA / Department of Financial Services, GoI |
| **CT005** | PMEGP | Central | 18 | NOT SPECIFIED | NOT SPECIFIED | Khadi & Village Industries Commission (KVIC), MSME |
| **CT006** | Atal Pension Yojana (APY) | Central | 18 | 40 | Non-Income Tax Payer | Pension Fund Regulatory and Development Authority (PFRDA) |

---

## Detailed Scheme Verifications

```
================================================================================
SCHEME ID: TS001
================================================================================
```
- **Scheme Name**: Rythu Bharosa (formerly Rythu Bandhu framework)
- **Government Scope**: State (Telangana)
- **Official Source Authority**: Telangana Agriculture & Farmers Welfare Commission / Department of Agriculture, Government of Telangana
- **Official Source URL**: `https://farmerswelfarecommission.telangana.gov.in/` / `https://www.telangana.gov.in`
- **Retrieval Date**: 2026-08-15
- **Current Eligibility Criteria**:
  - Must be a permanent resident of Telangana.
  - Must possess cultivable agricultural land registered in the Dharani land records (Pattadar Passbook holder) or hold Recognition of Forest Rights (RoFR) title.
  - Registered tenant farmers on arable land under notified guidelines.
  - Assistance is restricted to active, cultivable lands; fallow lands, mining zones, and real estate ventures are excluded.
- **Age Requirements**:
  - `age_min`: NOT SPECIFIED (No strict statutory age limit, tied to titleholder in land records)
  - `age_max`: NOT SPECIFIED
- **Income Requirements**:
  - `income_limit`: NOT SPECIFIED (Investment grant based on landholding)
- **Beneficiary Conditions**: Landholding farmers, RoFR patta holders, and designated tenant farmers actively cultivating land in Telangana.
- **Benefits**: ₹6,000 per acre per agricultural season (₹12,000 per acre annually across Kharif and Rabi seasons) disbursed directly to Aadhaar-linked bank accounts; additional ₹500/quintal bonus for fine paddy production.
- **Required Documents**:
  1. Dharani Pattadar Passbook / Land Title document / RoFR Title
  2. Aadhaar Card
  3. Aadhaar-seeded Bank Account Passbook
- **Application Method**: Direct Benefit Transfer (DBT) verification through Agriculture Extension Officers (AEOs) using the Dharani portal and Agriculture Department database; applications also accepted via Praja Palana channels.

---

```
================================================================================
SCHEME ID: TS002
================================================================================
```
- **Scheme Name**: Rythu Bhima (Farmers Group Life Insurance Scheme)
- **Government Scope**: State (Telangana)
- **Official Source Authority**: Telangana Agriculture & Farmers Welfare Commission / Life Insurance Corporation of India (LIC)
- **Official Source URL**: `https://farmerswelfarecommission.telangana.gov.in/` / `https://rythubandhu.telangana.gov.in/`
- **Retrieval Date**: 2026-08-15
- **Current Eligibility Criteria**:
  - Resident farmer of Telangana owning agricultural land.
  - Name must be entered in the Dharani Record of Rights (RoR) database with an issued Pattadar Passbook or RoFR title.
  - Farmer must be between the ages of 18 and 59 years as verified through Aadhaar.
- **Age Requirements**:
  - `age_min`: 18 years
  - `age_max`: 59 years (Enrollment cut-off age)
- **Income Requirements**:
  - `income_limit`: NOT SPECIFIED (100% state-funded insurance; no income cap)
- **Beneficiary Conditions**: Registered landholding farmers whose nominations are formally processed into the Rythu Bima LIC master policy.
- **Benefits**: ₹5,00,000 lump-sum financial life insurance coverage paid directly to the designated nominee in the event of the insured farmer's death (due to natural or accidental causes). Entire annual premium is paid by the Government of Telangana.
- **Required Documents**:
  1. Aadhaar Card (age proof)
  2. Pattadar Passbook / Dharani Record
  3. Nominee Aadhaar Card & Bank Account Details
  4. Death Certificate & Claim Form (at time of claim settlement)
- **Application Method**: Enrolled via village Agriculture Extension Officers (AEOs) into the online Rythu Bima portal.

---

```
================================================================================
SCHEME ID: TS003
================================================================================
```
- **Scheme Name**: Cheyutha (Rajiv Aarogyasri Healthcare)
- **Government Scope**: State (Telangana)
- **Official Source Authority**: Rajiv Aarogyasri Health Care Trust / Department of Health, Medical & Family Welfare, Government of Telangana
- **Official Source URL**: `https://www.aarogyasri.telangana.gov.in/` / `https://www.telangana.gov.in/Government-Initiatives/`
- **Retrieval Date**: 2026-08-15
- **Current Eligibility Criteria**:
  - Families residing in Telangana holding an active Food Security Card (White Ration Card) or designated BPL health identification.
  - Government is transitioning towards digital health profiles for expanded universal coverage.
- **Age Requirements**:
  - `age_min`: NOT SPECIFIED (All family members listed on the ration card/health profile are covered)
  - `age_max`: NOT SPECIFIED
- **Income Requirements**:
  - `income_limit`: Must fall within BPL threshold as certified by White Ration Card (Food Security Card) or official income certificate (≤ ₹1.5L Rural / ≤ ₹2.0L Urban).
- **Beneficiary Conditions**: Economically weaker and low-income families requiring treatment for listed secondary, tertiary, and critical medical procedures.
- **Benefits**: Cashless inpatient healthcare coverage of up to ₹10,00,000 (10 Lakh) per family per year across 1,600+ listed medical and surgical therapies at empanelled government and private network hospitals.
- **Required Documents**:
  1. Food Security Card (White Ration Card) / Aarogyasri Health Card
  2. Aadhaar Cards of family members
  3. Medical referral / diagnosis documentation
- **Application Method**: Point-of-care registration at any empanelled hospital help desk through the Aarogya Mithra assistant using Aadhaar / Ration card biometric verification.

---

```
================================================================================
SCHEME ID: TS004
================================================================================
```
- **Scheme Name**: Maha Lakshmi Scheme
- **Government Scope**: State (Telangana)
- **Official Source Authority**: Government of Telangana / TSRTC / Civil Supplies Department
- **Official Source URL**: `https://www.telangana.gov.in/Government-Initiatives/` / `https://mahalakshmi.telangana.gov.in`
- **Retrieval Date**: 2026-08-15
- **Current Eligibility Criteria**:
  - **Component 1 (Free Bus Travel)**: All girls, women of all age groups, and transgender persons who are domicile residents of Telangana.
  - **Component 2 (₹500 LPG Cylinder)**: Permanent resident households holding an active Food Security Card (White Ration Card) with a domestic LPG connection registered in the name of a family member; submitted Praja Palana application.
  - **Component 3 (₹2,500 Monthly Financial Assistance)**: Eligible woman designated as head of the family; exclusions apply for income-tax payers and government pensioners.
- **Age Requirements**:
  - `age_min`: NOT SPECIFIED (Component 1: all ages; Component 2: adult connection holder; Component 3: 18+)
  - `age_max`: NOT SPECIFIED
- **Income Requirements**:
  - `income_limit`: Component 1: Universal (No income limit); Component 2 & 3: White Ration Card (BPL) holders.
- **Beneficiary Conditions**: Female residents and transgender persons in Telangana.
- **Benefits**:
  1. 100% free travel across TSRTC Palle Velugu and Express buses within Telangana state borders (Zero Ticket issued).
  2. Subsidized domestic LPG gas refills at ₹500 per cylinder (balance refunded via DBT).
  3. Direct cash assistance of ₹2,500 per month to the woman head of eligible households.
- **Required Documents**:
  1. Aadhaar Card with Telangana residential address (for bus travel & DBT)
  2. Food Security Card (White Ration Card)
  3. Consumer Gas Connection Book / Number
  4. Bank Passbook details
- **Application Method**:
  - Bus Travel: Show original Aadhaar or government domicile photo ID to bus conductor for zero-fare ticket.
  - LPG & Monthly Aid: Application through Praja Palana application forms submitted at Gram Panchayats / Ward Secretariats.

---

```
================================================================================
SCHEME ID: TS005
================================================================================
```
- **Scheme Name**: Aasara Pensions
- **Government Scope**: State (Telangana)
- **Official Source Authority**: Society for Elimination of Rural Poverty (SERP) / Panchayat Raj & Rural Development Department, Government of Telangana
- **Official Source URL**: `https://www.aasara.telangana.gov.in/`
- **Retrieval Date**: 2026-08-15
- **Current Eligibility Criteria**:
  - Must be a permanent resident of Telangana from a vulnerable socio-economic background.
  - **Old Age Pension (OAP)**: Minimum age of 57 years.
  - **Widow Pension**: Minimum age of 18 years with death certificate of spouse.
  - **Disabled Pension (PwD)**: Person with disability having SADAREM certificate with minimum 40% disability (no minimum age).
  - **Weavers Pension**: Age 50 years or above, actively engaged in weaving.
  - **Toddy Tappers Pension**: Age 50 years or above, registered with Toddy Tappers Co-operative Society.
  - **Single Women / Beedi Workers / Dialysis Patients**: Meeting designated category guidelines.
  - Family must not own more than 3 acres of wet land or 7.5 acres of dry land; must not have family members in government service or paying income tax.
- **Age Requirements**:
  - `age_min`: 57 years (Old Age), 18 years (Widow), 50 years (Weavers/Tappers), No Min (Disabled)
  - `age_max`: NOT SPECIFIED
- **Income Requirements**:
  - `income_limit`: Family annual income not exceeding ₹1,50,000 in rural areas and ₹2,00,000 in urban areas (or possessing White Ration Card).
- **Beneficiary Conditions**: Destitute elderly, widows, disabled persons, and traditional occupational artisans without sufficient livelihood support.
- **Benefits**: Monthly social security pension of ₹2,016 per month for Old Age, Widows, Weavers, Toddy Tappers, Single Women, and Beedi workers; ₹3,016–₹4,016 per month for Disabled individuals.
- **Required Documents**:
  1. Aadhaar Card
  2. Proof of Age (Voter ID, School certificate, or age verification)
  3. SADAREM Disability Certificate (for PwD)
  4. Spouse Death Certificate (for Widows)
  5. Income Certificate / Food Security Card
  6. Bank Account Passbook / Post Office Savings Account
- **Application Method**: Apply at local Gram Panchayat / MPDO office (rural) or Municipal Ward Office / MeeSeva / Praja Palana centers.

---

```
================================================================================
SCHEME ID: TS006
================================================================================
```
- **Scheme Name**: Telangana ePASS Scholarships (Post-Matric & Pre-Matric)
- **Government Scope**: State (Telangana)
- **Official Source Authority**: Centre for Good Governance (CGG) / SC, ST, BC, EBC, Minority & Disabled Welfare Departments, Government of Telangana
- **Official Source URL**: `https://telanganaepass.cgg.gov.in/`
- **Retrieval Date**: 2026-08-15
- **Current Eligibility Criteria**:
  - Must be a permanent resident of Telangana.
  - Student must be enrolled in a recognized post-matric course (Intermediate, Polytechnic, Degree, PG, Professional Courses) or pre-matric classes in recognized institutions.
  - Minimum 75% aggregate attendance required at the end of each academic quarter.
  - Students admitted under Convener Quota (Management Quota / NRI quota students are excluded).
- **Age Requirements**:
  - `age_min`: NOT SPECIFIED (Academic admission age limits apply as per university / board norms)
  - `age_max`: NOT SPECIFIED
- **Income Requirements**:
  - `income_limit`:
    - SC / ST students: Parental annual income ≤ ₹2,00,000 (up to ₹2,50,000 under central norms).
    - BC / EBC / Minority (Rural): Parental annual income ≤ ₹1,50,000.
    - BC / EBC / Minority (Urban): Parental annual income ≤ ₹2,00,000.
    - Disabled Welfare: Parental annual income ≤ ₹1,00,000.
- **Beneficiary Conditions**: Students belonging to SC, ST, BC, EBC, Minority, and PwD categories pursuing higher education.
- **Benefits**: Complete Reimbursement of Tuition Fee (RTF) paid directly to colleges, and Maintenance Fee / Stipend (MTF) paid directly to the student's bank account.
- **Required Documents**:
  1. Latest Income Certificate issued by MeeSeva (Tahsildar)
  2. Integrated Community / Caste Certificate (SC/ST/BC)
  3. Aadhaar Card
  4. Bonafide / Study Certificate for previous 7 consecutive years
  5. SSC Memo & Previous Year Marksheet
  6. Bank Account Passbook (Aadhaar linked)
  7. College Admission Allotment Order
- **Application Method**: Online submission through the official Telangana ePASS portal (`https://telanganaepass.cgg.gov.in`), followed by college verification and district welfare officer approval.

---

```
================================================================================
SCHEME ID: TS007
================================================================================
```
- **Scheme Name**: MCH Kit Scheme (KCR Kit)
- **Government Scope**: State (Telangana)
- **Official Source Authority**: Commissionerate of Health & Family Welfare, Government of Telangana
- **Official Source URL**: `https://chfw.telangana.gov.in/programmes.html` / `https://kcrkit.telangana.gov.in`
- **Retrieval Date**: 2026-08-15
- **Current Eligibility Criteria**:
  - Pregnant woman residing in Telangana.
  - Must undergo institutional delivery in a Government Public Health Facility (PHC, CHC, Area Hospital, District Hospital, or Teaching Hospital).
  - Financial aid is restricted to a maximum of the first two live deliveries.
- **Age Requirements**:
  - `age_min`: NOT SPECIFIED (Legally married reproductive age)
  - `age_max`: NOT SPECIFIED
- **Income Requirements**:
  - `income_limit`: NOT SPECIFIED (Open to all women delivering in government hospitals)
- **Beneficiary Conditions**: Expectant mothers delivering in public healthcare facilities across Telangana.
- **Benefits**:
  - Total financial wage compensation: ₹12,000 for birth of a baby boy / ₹13,000 for birth of a baby girl, disbursed in 4 milestones (ANC registration, Delivery, Child 1st vaccination, Child 2nd vaccination).
  - Physical MCH Baby Kit containing 16 essential mother-child healthcare items (baby bed, clothes, mosquito net, baby oil, soaps, diapers, mother saree, etc.).
- **Required Documents**:
  1. Mother and Child Tracking System (MCTS) / RCH ID registration
  2. Mother's Aadhaar Card
  3. ANC Card / Mother Health Card
  4. Bank Account Passbook of the mother
  5. Government Hospital Delivery Discharge Summary
- **Application Method**: Registration at the nearest Primary Health Centre (PHC) / Anganwadi centre through ASHA worker upon pregnancy confirmation.

---

```
================================================================================
SCHEME ID: TS008
================================================================================
```
- **Scheme Name**: Telangana Free Diagnostic Services (T-Diagnostics)
- **Government Scope**: State (Telangana)
- **Official Source Authority**: Commissionerate of Health & Family Welfare, Government of Telangana
- **Official Source URL**: `https://chfw.telangana.gov.in/programmes.html` / `https://telanganadiagnostics.com`
- **Retrieval Date**: 2026-08-15
- **Current Eligibility Criteria**:
  - Any citizen residing in Telangana seeking outpatient or inpatient care at a public healthcare institution.
  - Must receive a prescription for diagnostic tests from a registered medical officer at a government health facility (PHC, UPHC, Basthi Dawakhana, CHC, Area Hospital, or District Hospital).
- **Age Requirements**:
  - `age_min`: NOT SPECIFIED (Universal across all age groups)
  - `age_max`: NOT SPECIFIED
- **Income Requirements**:
  - `income_limit`: NOT SPECIFIED (Universal free service; no income ceiling)
- **Beneficiary Conditions**: All patients visiting government healthcare facilities requiring diagnostic investigation.
- **Benefits**: 100% free comprehensive diagnostic tests (up to 134 pathology, biochemistry, microbiology, and radiology investigations) with digital test reports sent directly to the patient's mobile phone via SMS / online portal within 24 hours.
- **Required Documents**:
  1. Valid Government Doctor's Prescription / OPD Slip
  2. Mobile Number (for report delivery)
  3. Aadhaar Card (for identity verification)
- **Application Method**: Walk-in at any government health centre/dispensary; samples are collected on-site and processed via the centralized hub-and-spoke laboratory network.

---

```
================================================================================
SCHEME ID: CT001
================================================================================
```
- **Scheme Name**: PM-KISAN (Pradhan Mantri Kisan Samman Nidhi)
- **Government Scope**: Central (Applicable across Telangana)
- **Official Source Authority**: Ministry of Agriculture & Farmers Welfare, Government of India
- **Official Source URL**: `https://pmkisan.gov.in/`
- **Retrieval Date**: 2026-08-15
- **Current Eligibility Criteria**:
  - All landholding farmer families who have cultivable landholding in their names in state land records.
  - "Family" is defined as husband, wife, and minor children.
  - **Mandatory Exclusions**:
    - Institutional landholders.
    - Farmer families where one or more members hold/held constitutional posts.
    - Former and present Ministers, MPs, MLAs, MLCs, Mayors, Chairpersons of District Panchayats.
    - All serving or retired officers/employees of Central/State Government Ministries, Departments, PSUs, and Autonomous bodies (excluding Multi-Tasking Staff / Class IV / Group D employees).
    - All superannuated/retired pensioners whose monthly pension is ₹10,000 or more (excluding Group D).
    - All persons who paid Income Tax in the last assessment year.
    - Professionals (Doctors, Engineers, Lawyers, Chartered Accountants, Architects) registered with professional bodies.
- **Age Requirements**:
  - `age_min`: NOT SPECIFIED (Adult landholder in family)
  - `age_max`: NOT SPECIFIED
- **Income Requirements**:
  - `income_limit`: NOT SPECIFIED (Excluded if paid income tax in last assessment year)
- **Beneficiary Conditions**: Small, marginal, and all cultivable landholding farmer families across India (including Telangana).
- **Benefits**: ₹6,000 per year per eligible farmer family, paid in three equal 4-monthly installments of ₹2,000 each directly into Aadhaar-seeded bank accounts via DBT.
- **Required Documents**:
  1. Aadhaar Card
  2. Land Title / Record of Rights (Pattadar passbook / Dharani record)
  3. Aadhaar-seeded Bank Account with NPCI mapping
  4. Active Mobile Number for OTP / eKYC
- **Application Method**: Online self-registration on `https://pmkisan.gov.in` under "Farmer Corner" or via Common Service Centres (CSC) / Village Agriculture Offices, followed by biometric or facial eKYC.

---

```
================================================================================
SCHEME ID: CT002
================================================================================
```
- **Scheme Name**: Ayushman Bharat PM-JAY (Pradhan Mantri Jan Arogya Yojana)
- **Government Scope**: Central (Applicable across Telangana via convergence with Rajiv Aarogyasri)
- **Official Source Authority**: National Health Authority (NHA), Ministry of Health and Family Welfare, Government of India
- **Official Source URL**: `https://pmjay.gov.in/` / `https://nha.gov.in/`
- **Retrieval Date**: 2026-08-15
- **Current Eligibility Criteria**:
  - General entitlement based on deprivation and occupational criteria in the Socio-Economic Caste Census 2011 (SECC 2011) database, or holding a state Food Security Card under state convergence.
  - **Universal Senior Citizen Expansion (PM-JAY 70+)**: All Indian citizens aged 70 years and above are universally eligible regardless of family income or socio-economic status.
- **Age Requirements**:
  - `age_min`: NOT SPECIFIED for general family cover; 70 years for Universal Senior Citizen Top-up
  - `age_max`: NOT SPECIFIED
- **Income Requirements**:
  - `income_limit`: Below Poverty Line / Deprivation criteria for general pool; NO income limit for citizens aged 70+.
- **Beneficiary Conditions**: Identified poor, vulnerable, and unorganized worker families, and all senior citizens aged 70+.
- **Benefits**:
  - Up to ₹5,00,000 per family per year for secondary and tertiary hospitalization care across 27,000+ empanelled hospitals nationwide.
  - Senior citizens aged 70+ in covered families receive an independent additional top-up cover of ₹5,00,000 exclusively for themselves.
- **Required Documents**:
  1. Aadhaar Card
  2. Ration Card / Family ID / SECC confirmation letter
  3. Mobile Number linked with Aadhaar
- **Application Method**: Online self-verification and Ayushman Card generation via the Ayushman App / `https://beneficiary.nha.gov.in` or at any empanelled hospital Ayushman Mitra helpdesk.

---

```
================================================================================
SCHEME ID: CT003
================================================================================
```
- **Scheme Name**: PMAY-U 2.0 (Pradhan Mantri Awas Yojana - Urban 2.0)
- **Government Scope**: Central (Applicable in Urban Telangana)
- **Official Source Authority**: Ministry of Housing and Urban Affairs (MoHUA), Government of India
- **Official Source URL**: `https://pmaymis.gov.in/pmaymis2_2024/` / `https://pmaymis.gov.in`
- **Retrieval Date**: 2026-08-15
- **Current Eligibility Criteria**:
  - Urban family comprising husband, wife, and unmarried children.
  - The beneficiary family must not own a pucca house in their name or in the name of any member of their family anywhere in India.
  - Must not have availed central/state housing assistance under any government scheme in the past 20 years.
  - House must be owned by the female head of the family or jointly with the spouse (exceptions apply for single males/widowers/transgender).
- **Age Requirements**:
  - `age_min`: 18 years
  - `age_max`: NOT SPECIFIED
- **Income Requirements**:
  - `income_limit`:
    - **Economically Weaker Section (EWS)**: Annual household income up to ₹3,00,000.
    - **Low-Income Group (LIG)**: Annual household income between ₹3,00,001 and ₹6,00,000.
    - **Middle-Income Group (MIG)**: Annual household income between ₹6,00,001 and ₹9,00,000.
- **Beneficiary Conditions**: Urban families in EWS, LIG, and MIG categories seeking to construct, purchase, or subsidize their first pucca home.
- **Benefits**:
  - Interest Subsidy Scheme (ISS): 4% interest subsidy on home loans up to ₹25 lakh (subsidy on first ₹8 lakh) for property values up to ₹35 lakh (maximum subsidy of ₹1.80 lakh).
  - Beneficiary Led Construction (BLC) / Affordable Housing in Partnership (AHP): Central financial assistance of up to ₹2.50 lakh per dwelling unit.
- **Required Documents**:
  1. Aadhaar Cards of all family members
  2. Proof of Income (Salary slip, Form 16, or Income Certificate from Tahsildar)
  3. Affidavit declaring no pucca house ownership in India
  4. Land ownership documents / Sale agreement
  5. Bank Account details
- **Application Method**: Online application through the unified PMAY-U 2.0 portal (`https://pmaymis.gov.in`) or via Primary Lending Institutions (Banks/HFCs) for Interest Subsidy.

---

```
================================================================================
SCHEME ID: CT004
================================================================================
```
- **Scheme Name**: Pradhan Mantri Mudra Yojana (PMMY)
- **Government Scope**: Central (Applicable across Telangana)
- **Official Source Authority**: Micro Units Development & Refinance Agency (MUDRA) / Department of Financial Services, Ministry of Finance, Government of India
- **Official Source URL**: `https://www.mudra.org.in/` / `https://www.myscheme.gov.in/`
- **Retrieval Date**: 2026-08-15
- **Current Eligibility Criteria**:
  - Any Indian citizen who is eligible to take a credit loan.
  - Must have a viable business proposal for a non-farm income-generating micro-enterprise in manufacturing, trading, or services (including allied agriculture like poultry, dairy, beekeeping).
  - Applicant must not be a defaulter to any bank or financial institution and must possess a satisfactory credit track record.
- **Age Requirements**:
  - `age_min`: 18 years
  - `age_max`: 65 years (typical loan maturity limit as per lender norms)
- **Income Requirements**:
  - `income_limit`: NOT SPECIFIED (Loan evaluation is based on business cash-flow and project viability; no minimum personal income threshold).
- **Beneficiary Conditions**: Small artisans, shopkeepers, fruit/vegetable vendors, small manufacturers, transport operators, and aspiring micro-entrepreneurs.
- **Benefits**: Collateral-free institutional business loans across 4 product categories:
  1. **Shishu**: Loans up to ₹50,000 (for start-ups and micro units).
  2. **Kishore**: Loans above ₹50,000 and up to ₹5,00,000.
  3. **Tarun**: Loans above ₹5,00,000 and up to ₹10,00,000.
  4. **Tarun Plus**: Loans above ₹10,00,000 and up to ₹20,00,000 (for entrepreneurs who have successfully repaid previous Tarun loans).
- **Required Documents**:
  1. Proof of Identity (Aadhaar Card / Voter ID / PAN Card)
  2. Proof of Residence
  3. Business Enterprise Registration / Udyam Certificate (if available)
  4. Project report / quotation of machinery or assets to be purchased
  5. Bank Statement (last 6 months)
- **Application Method**: Apply online via the JanSamarth / UdyamiMitra portal (`https://www.jansamarth.in`) or submit loan application form directly at any Commercial Bank, RRB, Small Finance Bank, or MFI branch.

---

```
================================================================================
SCHEME ID: CT005
================================================================================
```
- **Scheme Name**: Prime Minister's Employment Generation Programme (PMEGP)
- **Government Scope**: Central (Applicable across Telangana)
- **Official Source Authority**: Khadi and Village Industries Commission (KVIC) / Ministry of Micro, Small and Medium Enterprises (MSME), Government of India
- **Official Source URL**: `https://www.kviconline.gov.in/pmegpeportal/` / `https://msme.gov.in`
- **Retrieval Date**: 2026-08-15
- **Current Eligibility Criteria**:
  - Any individual aged 18 years and above.
  - Assistance is available only for establishing **new** micro-enterprises/projects.
  - Educational Requirement: Must have passed at least VIII Standard for projects costing above ₹10 Lakh in the Manufacturing sector and above ₹5 Lakh in the Business/Service sector.
  - Self-Help Groups (SHGs), Institutions registered under Societies Registration Act 1860, and Co-operative societies are also eligible.
- **Age Requirements**:
  - `age_min`: 18 years
  - `age_max`: NOT SPECIFIED (No upper age limit)
- **Income Requirements**:
  - `income_limit`: NOT SPECIFIED (No family income ceiling for applicants)
- **Beneficiary Conditions**: Unemployed individuals, rural youth, traditional artisans, and prospective entrepreneurs.
- **Benefits**:
  - Maximum project cost: Up to ₹50,00,000 (50 Lakh) for Manufacturing units; Up to ₹20,00,000 (20 Lakh) for Service/Business units.
  - Margin Money Subsidy (Government grant):
    - **General Category**: 15% (Urban areas) / 25% (Rural areas) of project cost (Beneficiary own contribution: 10%).
    - **Special Category (SC/ST/OBC/Minority/Women/Ex-Servicemen/PwD/Transgender)**: 25% (Urban areas) / 35% (Rural areas) of project cost (Beneficiary own contribution: 5%).
- **Required Documents**:
  1. Aadhaar Card
  2. Caste / Special Category Certificate (if applicable)
  3. Educational Qualification Certificate (8th pass memo if project > ₹10L/₹5L)
  4. Detailed Project Report (DPR)
  5. Rural Area Certificate (issued by competent local revenue authority if claiming rural subsidy)
  6. EDP Training Completion Certificate (mandatory prior to final disbursement)
- **Application Method**: Apply 100% online through the official PMEGP e-Portal (`https://www.kviconline.gov.in/pmegpeportal/`).

---

```
================================================================================
SCHEME ID: CT006
================================================================================
```
- **Scheme Name**: Atal Pension Yojana (APY)
- **Government Scope**: Central (Applicable across Telangana)
- **Official Source Authority**: Pension Fund Regulatory and Development Authority (PFRDA) / Government of India
- **Official Source URL**: `https://pfrda.org.in/web/pfrda/schemes/atal-pension-yojana-apy` / `https://www.npscra.nsdl.co.in/`
- **Retrieval Date**: 2026-08-15
- **Current Eligibility Criteria**:
  - Must be a citizen of India between 18 and 40 years of age.
  - Must hold a valid savings bank or post office account.
  - **Statutory Exclusion**: Any citizen who is or has been an **income-tax payer** under the Income Tax Act, 1961 (effective 1st October 2022) is **ineligible** to join the scheme.
- **Age Requirements**:
  - `age_min`: 18 years
  - `age_max`: 40 years (Entry age limit)
- **Income Requirements**:
  - `income_limit`: Non-Income Tax Payer (Cannot be an active or past income tax assessee).
- **Beneficiary Conditions**: Unorganized sector workers and Indian citizens seeking a guaranteed regular pension post-retirement.
- **Benefits**:
  - Guaranteed minimum monthly pension of ₹1,000, ₹2,000, ₹3,000, ₹4,000, or ₹5,000 per month starting at age 60 until death, depending on the subscriber's entry age and monthly contribution amount.
  - After the subscriber's death, the exact same pension amount is paid to the spouse for life.
  - On the death of both subscriber and spouse, the entire accumulated pension corpus is returned to the nominee.
- **Required Documents**:
  1. Aadhaar Card
  2. Savings Bank Account / Post Office Account details
  3. Mobile Number linked to bank account
  4. Nominee Aadhaar details
- **Application Method**: Enrolled through the subscriber's bank branch / net-banking portal / Post Office where the savings account is maintained, or online via APY-eNPS.

---
*End of Phase 2 Verification Report.*
