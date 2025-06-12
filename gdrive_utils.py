import os
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import io
import pandas as pd
import tempfile
import re
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = os.getenv('GOOGLE_SERVICE_ACCOUNT_FILE', 'service_account.json')

def authenticate_gdrive():
    """Authenticate with Google Drive API using service account."""
    try:
        if not os.path.exists(SERVICE_ACCOUNT_FILE):
            raise FileNotFoundError(f"Service account file not found: {SERVICE_ACCOUNT_FILE}")
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES
        )
        service = build('drive', 'v3', credentials=credentials)
        logger.info("Google Drive authentication successful")
        return service
    except Exception as e:
        logger.error(f"Failed to authenticate: {str(e)}")
        try:
            import streamlit as st
            st.error(f"Google Drive authentication failed: {str(e)}")
        except ImportError:
            print(f"Error: Google Drive authentication failed: {str(e)}")
        return None

def create_gdrive_folder(folder_name, parent_id=None, service=None):
    """Create a folder in Google Drive."""
    try:
        if not service:
            service = authenticate_gdrive()
            if not service:
                return None
        
        folder_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_id] if parent_id else []
        }
        
        folder = service.files().create(
            body=folder_metadata,
            fields='id, name'
        ).execute()
        
        logger.info(f"Created folder: {folder.get('name')} (ID: {folder.get('id')})")
        try:
            import streamlit as st
            st.success(f"Created folder: {folder.get('name')}")
        except ImportError:
            print(f"Created folder: {folder.get('name')}")
        return folder.get('id')
    except Exception as e:
        logger.error(f"Folder creation failed: {str(e)}")
        try:
            import streamlit as st
            st.error(f"Folder creation failed: {str(e)}")
        except ImportError:
            print(f"Error: Folder creation failed: {str(e)}")
        return None

def upload_to_gdrive(file_path, folder_id=None, service=None, custom_filename=None):
    """Upload file to Google Drive with an optional custom filename."""
    try:
        if not service:
            service = authenticate_gdrive()
            if not service:
                return None
        
        file_name = custom_filename if custom_filename else os.path.basename(file_path)
        
        file_metadata = {
            'name': file_name,
            'parents': [folder_id] if folder_id else []
        }
        
        media = MediaFileUpload(file_path, resumable=True)
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, name, webViewLink, mimeType'
        ).execute()
        
        logger.info(f"Uploaded file: {file.get('name')} (ID: {file.get('id')})")
        try:
            import streamlit as st
            st.success(f"Uploaded file: {file.get('name')}")
        except ImportError:
            print(f"Uploaded file: {file.get('name')}")
        return file
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        try:
            import streamlit as st
            st.error(f"Upload failed: {str(e)}")
        except ImportError:
            print(f"Error: Upload failed: {str(e)}")
        return None

def download_from_gdrive(file_id, save_path=None, service=None):
    """Download file from Google Drive."""
    try:
        if not service:
            service = authenticate_gdrive()
            if not service:
                return None
        
        file_metadata = service.files().get(
            fileId=file_id,
            fields='name, mimeType'
        ).execute()
        
        file_name = file_metadata['name']
        
        if save_path and os.path.isdir(save_path):
            save_path = os.path.join(save_path, file_name)
        
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        try:
            import streamlit as st
            progress_bar = st.progress(0)
            status_text = st.empty()
        except ImportError:
            progress_bar = status_text = None
        
        while not done:
            status, done = downloader.next_chunk()
            if status and progress_bar:
                progress = int(status.progress() * 100)
                progress_bar.progress(progress)
                status_text.text(f"Downloading {file_name}... {progress}%")
        
        if progress_bar:
            progress_bar.empty()
            status_text.empty()
        
        if save_path:
            with open(save_path, 'wb') as f:
                f.write(fh.getbuffer())
            logger.info(f"File saved to: {save_path}")
            try:
                import streamlit as st
                st.success(f"File saved to: {save_path}")
            except ImportError:
                print(f"File saved to: {save_path}")
        
        return {
            'name': file_name,
            'content': fh.getvalue(),
            'mimeType': file_metadata['mimeType']
        }
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        try:
            import streamlit as st
            st.error(f"Download failed: {str(e)}")
        except ImportError:
            print(f"Error: Download failed: {str(e)}")
        return None

def update_metadata_csv(folder_id, metadata_entry, service=None, root_folder_id=None):
    """Update or create metadata.csv in the root folder."""
    try:
        if not service:
            service = authenticate_gdrive()
            if not service:
                return None
        
        results = service.files().list(
            q=f"'{root_folder_id}' in parents and name='metadata.csv' and mimeType='text/csv'",
            fields="files(id, name)"
        ).execute()
        files = results.get('files', [])
        
        metadata_file_id = files[0]['id'] if files else None
        
        new_entry = pd.DataFrame([metadata_entry])
        
        if metadata_file_id:
            existing_data = download_from_gdrive(metadata_file_id, service=service)
            if existing_data:
                existing_df = pd.read_csv(io.BytesIO(existing_data['content']))
                updated_df = pd.concat([existing_df, new_entry], ignore_index=True)
            else:
                updated_df = new_entry
        else:
            updated_df = new_entry
        
        temp_dir = os.path.join(tempfile.gettempdir(), 'gdrive_temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_csv_path = os.path.join(temp_dir, 'metadata.csv')
        updated_df.to_csv(temp_csv_path, index=False, encoding='utf-8-sig')
        
        if metadata_file_id:
            file_metadata = {'name': 'metadata.csv'}
            media = MediaFileUpload(temp_csv_path, mimetype='text/csv')
            service.files().update(
                fileId=metadata_file_id,
                body=file_metadata,
                media_body=media,
                fields='id, name'
            ).execute()
            logger.info("Updated metadata.csv")
            try:
                import streamlit as st
                st.success("Updated metadata.csv")
            except ImportError:
                print("Updated metadata.csv")
        else:
            upload_to_gdrive(temp_csv_path, folder_id=root_folder_id, service=service)
        
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)
        
        return True
    except Exception as e:
        logger.error(f"Failed to update metadata.csv: {str(e)}")
        try:
            import streamlit as st
            st.error(f"Failed to update metadata.csv: {str(e)}")
        except ImportError:
            print(f"Error: Failed to update metadata.csv: {str(e)}")
        return False

def search_gdrive_files(query=None, min_lead=None, max_lead=None, min_intent=None, max_intent=None, service=None):
    """Search files in Google Drive with filters for lead and intent scores."""
    try:
        if not service:
            service = authenticate_gdrive()
            if not service:
                return []
        
        query_parts = []
        if query:
            query_parts.append(f"name contains '{query}'")
        
        full_query = " and ".join(query_parts) if query_parts else ""
        
        results = service.files().list(
            q=full_query,
            pageSize=100,
            fields="nextPageToken, files(id, name, webViewLink, createdTime, mimeType, size)",
            orderBy="createdTime desc"
        ).execute()
        
        items = results.get('files', [])
        
        if not items:
            logger.info("No files found")
            try:
                import streamlit as st
                st.warning("No files found")
            except ImportError:
                print("No files found")
            return []
        
        filtered_items = []
        for item in items:
            filename = item['name']
            match = True
            
            lead_score = None
            intent_score = None
            lead_match = re.search(r'_L(\d+)', filename)
            intent_match = re.search(r'_I(\d+)', filename)
            
            if lead_match:
                lead_score = int(lead_match.group(1))
            if intent_match:
                intent_score = int(intent_match.group(1))
            
            if min_lead is not None and (lead_score is None or lead_score < min_lead):
                match = False
            if max_lead is not None and (lead_score is None or lead_score > max_lead):
                match = False
            if min_intent is not None and (intent_score is None or intent_score < min_intent):
                match = False
            if max_intent is not None and (intent_score is None or intent_score > max_intent):
                match = False
            
            if match:
                filtered_items.append(item)
        
        if not filtered_items:
            logger.info("No files matched criteria")
            try:
                import streamlit as st
                st.warning("No files matched criteria")
            except ImportError:
                print("No files matched criteria")
            return []
        
        return filtered_items
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        try:
            import streamlit as st
            st.error(f"Search failed: {str(e)}")
        except ImportError:
            print(f"Error: Search failed: {str(e)}")
        return []