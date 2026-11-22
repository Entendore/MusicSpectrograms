import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import re

class FileRenamerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bulk File Renamer")
        self.root.geometry("800x600")

        # Variable storage
        self.folder_path = tk.StringVar()
        self.replace_spaces = tk.BooleanVar(value=True)
        self.remove_special = tk.BooleanVar(value=True)
        self.to_lowercase = tk.BooleanVar(value=False)
        
        # --- Layout ---
        # Top Frame: Folder Selection
        frame_top = tk.Frame(root, padx=10, pady=10)
        frame_top.pack(fill=tk.X)

        tk.Label(frame_top, text="Folder:").pack(side=tk.LEFT)
        tk.Entry(frame_top, textvariable=self.folder_path, width=60).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        tk.Button(frame_top, text="Browse", command=self.browse_folder).pack(side=tk.LEFT)

        # Middle Frame: Options
        frame_options = tk.LabelFrame(root, text="Renaming Options", padx=10, pady=10)
        frame_options.pack(fill=tk.X, padx=10, pady=5)

        tk.Checkbutton(frame_options, text="Replace spaces with underscores (_)", variable=self.replace_spaces).pack(anchor=tk.W)
        tk.Checkbutton(frame_options, text="Remove special characters (keep letters, numbers, dashes, dots)", variable=self.remove_special).pack(anchor=tk.W)
        tk.Checkbutton(frame_options, text="Convert filename to lowercase", variable=self.to_lowercase).pack(anchor=tk.W)

        # Action Buttons
        frame_actions = tk.Frame(root, padx=10, pady=5)
        frame_actions.pack(fill=tk.X)
        
        tk.Button(frame_actions, text="Preview Changes", command=self.preview_changes, bg="#dddddd").pack(side=tk.LEFT, padx=5)
        tk.Button(frame_actions, text="Apply Changes", command=self.apply_changes, bg="#90ee90", fg="black").pack(side=tk.LEFT, padx=5)

        # Bottom Frame: Preview List
        frame_list = tk.Frame(root, padx=10, pady=10)
        frame_list.pack(fill=tk.BOTH, expand=True)

        # Treeview for preview
        cols = ("Original", "New")
        self.tree = ttk.Treeview(frame_list, columns=cols, show="headings")
        self.tree.heading("Original", text="Original Name")
        self.tree.heading("New", text="New Name")
        self.tree.column("Original", width=350)
        self.tree.column("New", width=350)

        # Scrollbars
        vsb = ttk.Scrollbar(frame_list, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

    def browse_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.folder_path.set(folder_selected)
            self.preview_changes()

    def clean_filename(self, filename):
        # Separate name and extension
        name, ext = os.path.splitext(filename)
        
        # 1. Remove special characters first
        # Keeps letters, numbers, spaces, dashes, dots, and underscores
        if self.remove_special.get():
            name = re.sub(r"[^\w\s\-\.]", "", name)

        # 2. Normalize "Space Dash Space" to just "Dash"
        # This prevents "File - Name" from becoming "File_-_Name"
        # It turns " - " into "-"
        name = re.sub(r"\s*-\s*", "-", name)

        # 3. Replace spaces with underscores
        if self.replace_spaces.get():
            name = name.replace(" ", "_")

        # 4. Lowercase
        if self.to_lowercase.get():
            name = name.lower()
            ext = ext.lower()

        # 5. Final Cleanup
        # Fix ugly sequences like "_-_", "__", "--"
        # If the sequence contains a dash, reduce to a single dash. Else, reduce to underscore.
        def clean_separator_match(match):
            s = match.group()
            if '-' in s:
                return '-'
            return '_'

        # Matches any sequence of 2 or more underscores/dashes
        name = re.sub(r"[_\-]{2,}", clean_separator_match, name)

        # Remove leading/trailing separators
        name = name.strip("_- ")
        
        # Safety: Prevent empty filenames
        if not name:
            name = "file"

        return name + ext

    def get_rename_map(self):
        """Returns a list of tuples (old_name, new_name, full_old_path, full_new_path)"""
        folder = self.folder_path.get()
        if not folder or not os.path.isdir(folder):
            return []

        rename_list = []
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        
        for filename in files:
            new_filename = self.clean_filename(filename)
            
            if new_filename == filename:
                continue
                
            old_path = os.path.join(folder, filename)
            new_path = os.path.join(folder, new_filename)
            
            rename_list.append((filename, new_filename, old_path, new_path))
            
        return rename_list

    def preview_changes(self):
        # Clear existing list
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        rename_list = self.get_rename_map()
        
        if not rename_list:
            if self.folder_path.get():
                messagebox.showinfo("Info", "No files need renaming based on current options.")
            return

        for item in rename_list:
            self.tree.insert("", tk.END, values=(item[0], item[1]))

    def apply_changes(self):
        rename_list = self.get_rename_map()
        
        if not rename_list:
            messagebox.showinfo("Info", "No files to rename.")
            return

        # Confirmation with specific warning about overwriting
        confirm = messagebox.askyesno(
            "Confirm Overwrite", 
            f"{len(rename_list)} files will be renamed.\n\n"
            "WARNING: If a target filename already exists,\n"
            "the existing file will be DELETED (overwritten).\n\n"
            "Do you want to continue?"
        )
        if not confirm:
            return

        success_count = 0
        error_count = 0
        
        # Sort list to handle renaming chains safely
        rename_list.sort(key=lambda x: x[0])

        for old_name, new_name, old_path, new_path in rename_list:
            try:
                # Handle duplicate filenames / collisions
                if os.path.exists(new_path):
                    # Handle case-insensitive systems (Windows) where File.txt -> file.txt
                    if os.path.normcase(old_path) == os.path.normcase(new_path):
                        # Rename to temp first
                        temp_path = new_path + ".tmp_rename_placeholder"
                        os.rename(old_path, temp_path)
                        old_path = temp_path
                    else:
                        # Delete the duplicate target
                        os.remove(new_path)
                
                os.rename(old_path, new_path)
                success_count += 1
            except Exception as e:
                error_count += 1
                print(f"Error renaming {old_name}: {e}")

        msg = f"Process Complete.\nRenamed: {success_count}\nErrors: {error_count}"
        messagebox.showinfo("Result", msg)
        
        # Refresh the list
        self.preview_changes()

if __name__ == "__main__":
    root = tk.Tk()
    app = FileRenamerApp(root)
    root.mainloop()