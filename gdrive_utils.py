import os
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Google Drive API setup
SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = 'service_account.json'

def authenticate_gdrive():
    """Authenticate with Google Drive using service account"""
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)

def upload_to_gdrive(file_path, folder_id=None, service=None):
    """Upload file to Google Drive"""
    try:
        if not service:
            service = authenticate_gdrive()
        
        file_metadata = {
            'name': os.path.basename(file_path),
            'parents': [folder_id] if folder_id else []
        }
        
        media = MediaFileUpload(file_path, resumable=True)
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, name, webViewLink'
        ).execute()
        
        print(f"✅ Uploaded to Google Drive: {file.get('name')}")
        print(f"🔗 View file at: {file.get('webViewLink')}")
        return file
    except Exception as e:
        print(f"❌ Google Drive upload failed: {str(e)}")
        return None

def search_gdrive_files(query=None, min_lead=None, max_lead=None, 
                       min_intent=None, max_intent=None, 
                       start_date=None, end_date=None, service=None):
    """Search files in Google Drive with filters"""
    try:
        if not service:
            service = authenticate_gdrive()
        
        # Base query parts
        query_parts = []
        if query:
            query_parts.append(f"name contains '{query}'")
        
        # Date range filtering in query if provided
        if start_date or end_date:
            date_query = []
            if start_date:
                date_query.append(f"createdTime >= '{start_date}T00:00:00'")
            if end_date:
                date_query.append(f"createdTime <= '{end_date}T23:59:59'")
            query_parts.append(" and ".join(date_query))
        
        # Get all files first (we'll filter locally for scores)
        full_query = " and ".join(query_parts) if query_parts else ""
        
        results = service.files().list(
            q=full_query,
            pageSize=100,  # Increased to get more results for local filtering
            fields="nextPageToken, files(id, name, webViewLink, createdTime)",
            orderBy="createdTime desc"  # Sort by creation date by default
        ).execute()
        
        items = results.get('files', [])
        
        if not items:
            print("No files found matching your criteria.")
            return []
        
        # Filter items based on score ranges if specified
        filtered_items = []
        for item in items:
            filename = item['name']
            match = True
            
            # Extract lead score if needed
            lead_score = None
            if min_lead is not None or max_lead is not None:
                if 'Lead Score: ' in filename:
                    try:
                        lead_part = filename.split('Lead Score: ')[1]
                        lead_score = int(lead_part.split()[0].rstrip(';,'))
                    except (IndexError, ValueError):
                        pass
                elif '_L' in filename:  # Fallback to old format if needed
                    try:
                        lead_part = filename.split('_L')[1]
                        lead_score = int(lead_part.split('_')[0])
                    except (IndexError, ValueError):
                        pass
            
            # Check lead score range
            if min_lead is not None and lead_score is not None and lead_score < min_lead:
                match = False
            if max_lead is not None and lead_score is not None and lead_score > max_lead:
                match = False
            
            # Extract intent score if needed
            intent_score = None
            if min_intent is not None or max_intent is not None:
                if 'Intent Score: ' in filename:
                    try:
                        intent_part = filename.split('Intent Score: ')[1]
                        intent_score = int(intent_part.split()[0].rstrip(';,'))
                    except (IndexError, ValueError):
                        pass
                elif '_I' in filename:  # Fallback to old format if needed
                    try:
                        intent_part = filename.split('_I')[1]
                        intent_score = int(intent_part.split('_')[0])
                    except (IndexError, ValueError):
                        pass
            
            # Check intent score range
            if min_intent is not None and intent_score is not None and intent_score < min_intent:
                match = False
            if max_intent is not None and intent_score is not None and intent_score > max_intent:
                match = False
            
            # Extract date from filename if "Created: " exists
            file_date = None
            if 'Created: ' in filename:
                try:
                    date_part = filename.split('Created: ')[1].split()[0]
                    file_date = datetime.strptime(date_part, '%Y-%m-%d').date()
                except (IndexError, ValueError):
                    pass
            
            # If date filtering is needed but no "Created: " in filename, use createdTime
            if (start_date or end_date) and file_date is None:
                try:
                    created_time = datetime.strptime(item['createdTime'], '%Y-%m-%dT%H:%M:%S.%fZ')
                    file_date = created_time.date()
                except (KeyError, ValueError):
                    pass
            
            # Check date range if needed
            if start_date and file_date:
                if file_date < datetime.strptime(start_date, '%Y-%m-%d').date():
                    match = False
            if end_date and file_date:
                if file_date > datetime.strptime(end_date, '%Y-%m-%d').date():
                    match = False
            
            if match:
                filtered_items.append(item)
        
        if not filtered_items:
            print("No files found matching all your criteria.")
            return []
        
        print(f"Found {len(filtered_items)} files:")
        for item in filtered_items:
            print(f"\n📄 {item['name']}")
            print(f"🕒 Created: {item['createdTime']}")
            print(f"🔗 View: {item['webViewLink']}")
        
        return filtered_items
    except Exception as e:
        print(f"❌ Google Drive search failed: {str(e)}")
        return []

def search_menu():
    """Interactive search menu"""
    while True:
        print("\n=== Google Drive Search ===")
        print("1. Search by filename")
        print("2. Filter by lead score range")
        print("3. Filter by intent score range")
        print("4. Search by date range")
        print("5. Combined search")
        print("6. Back to main menu")
        
        choice = input("Enter your choice: ").strip()
        
        if choice == "1":
            query = input("Enter search term: ").strip()
            search_gdrive_files(query=query)
        elif choice == "2":
            min_lead = int(input("Minimum lead score (0-100): "))
            max_lead = int(input("Maximum lead score (0-100): "))
            search_gdrive_files(min_lead=min_lead, max_lead=max_lead)
        elif choice == "3":
            min_intent = int(input("Minimum intent score (0-100): "))
            max_intent = int(input("Maximum intent score (0-100): "))
            search_gdrive_files(min_intent=min_intent, max_intent=max_intent)
        elif choice == "4":
            start_date = input("Start date (YYYY-MM-DD, leave empty for no limit): ").strip() or None
            end_date = input("End date (YYYY-MM-DD, leave empty for no limit): ").strip() or None
            search_gdrive_files(start_date=start_date, end_date=end_date)
        elif choice == "5":
            query = input("Enter search term (leave empty to skip): ").strip() or None
            min_lead = int(input("Minimum lead score (0-100, leave empty to skip): ") or 0)
            max_lead = int(input("Maximum lead score (0-100, leave empty to skip): ") or 100)
            min_intent = int(input("Minimum intent score (0-100, leave empty to skip): ") or 0)
            max_intent = int(input("Maximum intent score (0-100, leave empty to skip): ") or 100)
            start_date = input("Start date (YYYY-MM-DD, leave empty to skip): ").strip() or None
            end_date = input("End date (YYYY-MM-DD, leave empty to skip): ").strip() or None
            search_gdrive_files(
                query=query,
                min_lead=min_lead,
                max_lead=max_lead,
                min_intent=min_intent,
                max_intent=max_intent,
                start_date=start_date,
                end_date=end_date
            )
        elif choice == "6":
            break
        else:
            print("Invalid choice, please try again.")