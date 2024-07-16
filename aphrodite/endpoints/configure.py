import subprocess
import sys
from blessed import Terminal
import signal
import os
from fuzzywuzzy import process

def configure(args):
    term = Terminal()

    class Field:
        def __init__(self, name, prompt, field_type, options=None,
                     validator=None, optional=True, default=None,
                     advanced=False):
            self.name = name
            self.prompt = prompt
            self.field_type = field_type
            self.options = options
            self.validator = validator
            self.optional = optional
            self.value = default
            self.advanced = advanced

    def validate_int(value):
        return lambda x: x.isdigit() or x == ""

    def validate_float(value):
        return lambda x: x.replace('.', '').isdigit() and 0 <= float(x) <= value or x == ""

    fields = [
        Field("model", "Model Name", "input", optional=False),
        Field("tensor_parallel_size", "Tensor Parallel Size", "input",
              validator=validate_int, optional=True),
        Field("gpu_memory_utilization", "GPU Memory Utilization (%)", "input", 
              validator=validate_float, optional=True),
        Field("enable_chunked_prefill", "Enable Chunked Prefill", "checkbox", optional=True),
        Field("revision", "Model Revision", "input", optional=True, advanced=True),
        Field("max_logprobs", "Max Logprobs", "input", validator=validate_int, optional=True, advanced=True),
        Field("trust_remote_code", "Trust Remote Code", "checkbox", optional=True, advanced=True),
        Field("dtype", "Data Type", "multichoice", options=["auto", "float16", "bfloat16", "float32"], optional=True, advanced=True),
        Field("enforce_eager", "Enforce Eager", "boolean", optional=True, default=True, advanced=True),
    ]

    def draw_ui(current_field, show_advanced):
        print(term.clear)
        print(term.bold + term.cyan + "Aphrodite Configuration" + term.normal)
        print(term.yellow + "Use ↑/↓ arrows to navigate, Space to toggle/cycle, Tab for completion" + term.normal)
        print(term.yellow + "Press F2 to toggle advanced options, Enter to finish, Esc or Ctrl+C to cancel" + term.normal)
        print()

        print(term.bold + "Required Fields:" + term.normal)
        for i, field in enumerate(fields):
            if not field.optional:
                draw_field(i, field, current_field)

        print()
        print(term.bold + "Optional Fields:" + term.normal)
        for i, field in enumerate(fields):
            if field.optional and not field.advanced:
                draw_field(i, field, current_field)

        if show_advanced:
            print()
            print(term.bold + "Advanced Options:" + term.normal)
            for i, field in enumerate(fields):
                if field.advanced:
                    draw_field(i, field, current_field)
        
        print("\n" + term.yellow + "Press Enter when finished to launch the engine" + term.normal)

    def draw_field(i, field, current_field):
        if i == current_field:
            print(term.bold + term.green, end="")
        
        if field.field_type == "checkbox":
            checkbox = "[x]" if field.value else "[ ]"
            print(f"{'>' if i == current_field else ' '} {checkbox} {field.prompt}" + term.normal)
        elif field.field_type == "boolean":
            value = "True" if field.value is True else "False" if field.value is False else "Default"
            print(f"{'>' if i == current_field else ' '} {field.prompt}: {value}" + term.normal)
        elif field.field_type == "multichoice":
            print(f"{'>' if i == current_field else ' '} {field.prompt}: {field.value or 'Not set'}" + term.normal)
        else:
            print(f"{'>' if i == current_field else ' '} {field.prompt}: {field.value or ''}", end="")
            if i == current_field:
                print(term.blue + "█" + term.normal)
            else:
                print(term.normal)

    def expand_user_path(path):
        return os.path.expanduser(path)

    def path_complete(path):
        path = expand_user_path(path)
        directory = os.path.dirname(path) or '.'
        filename = os.path.basename(path)
        
        try:
            files = os.listdir(directory)
        except OSError:
            return []

        if not filename.startswith('.'):
            files = [f for f in files if not f.startswith('.')]

        matches = [f for f in files if f.lower().startswith(filename.lower())]
        
        if matches:
            return [os.path.join(directory, sorted(matches)[0])]
        
        fuzzy_matches = process.extractOne(filename, files)
        if fuzzy_matches and fuzzy_matches[1] > 70:
            return [os.path.join(directory, fuzzy_matches[0])]
        
        return []

    current_field = 0
    show_advanced_options = False

    def signal_handler(sig, frame):
        print(term.normal + term.clear)
        print("Configuration cancelled.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        with term.cbreak(), term.hidden_cursor():
            while True:
                draw_ui(current_field, show_advanced_options)
                key = term.inkey()

                if key.name == 'KEY_ESCAPE':
                    raise KeyboardInterrupt
                elif key.name == 'KEY_TAB':
                    if current_field == 0:  # Model Name field
                        completions = path_complete(fields[0].value)
                        if completions:
                            fields[0].value = completions[0]
                            if os.path.isdir(completions[0]):
                                fields[0].value += os.path.sep
                elif key.name == 'KEY_ENTER':
                    if not fields[0].value:  # Ensure required field is filled
                        continue
                    break  # Finished entering data
                elif key.name == 'KEY_UP':
                    current_field = (current_field - 1) % len(fields)
                    while fields[current_field].advanced and not show_advanced_options:
                        current_field = (current_field - 1) % len(fields)
                elif key.name == 'KEY_DOWN':
                    current_field = (current_field + 1) % len(fields)
                    while fields[current_field].advanced and not show_advanced_options:
                        current_field = (current_field + 1) % len(fields)
                elif key == ' ':
                    field = fields[current_field]
                    if field.field_type == "checkbox":
                        field.value = not field.value
                    elif field.field_type == "boolean":
                        if field.value is True:
                            field.value = False
                        elif field.value is False:
                            field.value = None
                        else:
                            field.value = True
                    elif field.field_type == "multichoice":
                        if field.value is None:
                            field.value = field.options[0]
                        else:
                            index = (field.options.index(field.value) + 1) % len(field.options)
                            field.value = field.options[index]
                elif key.name == 'KEY_F2':
                    show_advanced_options = not show_advanced_options
                    if not show_advanced_options:
                        current_field = next((i for i, f in enumerate(fields) if not f.advanced), 0)
                elif key.name == 'KEY_BACKSPACE':
                    if not fields[current_field].field_type in ["checkbox", "boolean", "multichoice"]:
                        fields[current_field].value = fields[current_field].value[:-1] if fields[current_field].value else ""
                elif not key.is_sequence and fields[current_field].field_type == "input":
                    field = fields[current_field]
                    new_value = (field.value or "") + key
                    if not field.validator or field.validator(new_value):
                        field.value = new_value

    except KeyboardInterrupt:
        print(term.normal + term.clear)
        print("Configuration cancelled.")
        return

    # Construct and execute command
    command = f"aphrodite run {fields[0].value}"
    for field in fields[1:]:
        if field.value:
            if field.field_type == "checkbox" and field.value:
                command += f" --{field.name.replace('_', '-')}"
            elif field.field_type == "boolean" and field.value is not None:
                command += f" --{field.name.replace('_', '-')} {str(field.value).lower()}"
            elif field.field_type in ["input", "multichoice"]:
                command += f" --{field.name.replace('_', '-')} {field.value}"
            elif field.name == "gmu":
                command += f" -gmu {float(field.value) / 100:.2f}"

    print(term.clear)
    print(term.green + "Executing command:" + term.normal)
    print(command)
    print()

    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(term.red + f"Error executing command: {e}" + term.normal,
              file=sys.stderr)
        sys.exit(1)