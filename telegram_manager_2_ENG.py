import asyncio
import logging
import subprocess
import os
import io
import time
import json
from typing import Tuple # <--- ADDED AS REQUESTED
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, ContextTypes,
    ConversationHandler, CallbackQueryHandler
)
from dotenv import load_dotenv
import sys
import subprocess
# IMPORTANT: psutil is no longer needed for stopping, but might be useful elsewhere.
# If it is not used anywhere else, it can be removed.
import psutil 
import signal
import re
import html

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KEY_ENV_PATH = os.path.join(SCRIPT_DIR, "key.env")

load_dotenv(dotenv_path=KEY_ENV_PATH)

def find_and_set_existing_process():
    """Finds an existing orchestrator process at startup and sets it to the global variable."""
    global trading_bot_process
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Check if our script name is in the command line
            if TRADING_BOT_SCRIPT_NAME in ' '.join(proc.info['cmdline']):
                logger.info(f"Found existing orchestrator process with PID: {proc.info['pid']}. Resuming control...")
                # Create a Popen object from the existing process (simulated)
                # trading_bot_process = subprocess.Popen.send_signal(subprocess.signal.CTRL_C_EVENT, proc.info['pid']) # Commented out in original
                trading_bot_process = psutil.Process(proc.info['pid'])
                # psutil.Process has similar methods to Popen (poll, terminate, kill)
                # Important: we cannot fully restore the Popen object, but we can manage the process via psutil
                return
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, TypeError):
            pass
    logger.info("No active orchestrator process found.")


def tail(filepath, n_lines):
    """Effectively reads the last 'n_lines' from a file without loading it entirely into memory."""
    try:
        with open(filepath, "rb") as f:
            f.seek(0, os.SEEK_END)
            end_byte = f.tell()
            lines_to_go = n_lines
            block_size = -1024
            blocks = []
            while lines_to_go > 0 and end_byte > 0:
                if (end_byte + block_size < 0):
                    block_size = -end_byte
                f.seek(block_size, os.SEEK_CUR)
                block_data = f.read(-block_size)
                blocks.append(block_data.decode('utf-8', errors='ignore'))
                lines_to_go -= blocks[-1].count('\n')
                end_byte += block_size
            all_lines_str = "".join(reversed(blocks))
            return all_lines_str.splitlines()[-n_lines:]
    except (FileNotFoundError, io.UnsupportedOperation):
        return []
    except Exception as e:
        logger.error(f"Error in tail function: {e}")
        return []
    
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- PATHS & CONSTANTS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRADING_BOT_SCRIPT_NAME = "new_logic_2_ENG.py"
TRADING_BOT_SCRIPT = os.path.join(SCRIPT_DIR, TRADING_BOT_SCRIPT_NAME)

# --- RESTORED OLD FLAGS ---
GLOBAL_STOP_FLAG_FILE = os.path.join(SCRIPT_DIR, "STOP_BOT_NOW.flag")
ACTIVE_BOT_CONFIG_FILE = os.path.join(SCRIPT_DIR, "active_bots_config.json")
RELOAD_CONFIG_FLAG_MANAGER = os.path.join(SCRIPT_DIR, "RELOAD_CONFIG.flag")

# --- RESTORED GLOBAL VARIABLE FOR PROCESS TRACKING ---
trading_bot_process = None
training_process_info = {}
# --- Conversation States ---
(
    # Old states (0-5)
    MANAGE_BOT_CONFIG_MENU, ASK_SYMBOL, ASK_CAPITAL, ASK_THRESHOLD,
    SHOW_GLOBAL_PARAMS_MENU, AWAIT_GLOBAL_PARAM_VALUE_INPUT,
    
    # New states for training dialog (6-14)
    SHOW_TRAINING_OPTIONS,      # 6: (Train / Validate / 80-20)
    
    TRAIN_ONLY_ASK_SYMBOL,      # 7: Select symbol for --train-only
    TRAIN_ONLY_ASK_START,       # 8: Await start date
    TRAIN_ONLY_ASK_END,         # 9: Await end date
    
    VALIDATE_ONLY_ASK_SYMBOL,   # 10: Select symbol for --validate-only
    VALIDATE_ONLY_ASK_START,    # 11: Await start date
    VALIDATE_ONLY_ASK_END,      # 12: Await end date
    
    TRAIN_8020_ASK_SYMBOL       # 13: Select symbol for --train-and-validate (old logic)
) = range(14)


# --- Button Texts ---
BUTTON_TEXT_APPLY_CONFIG_START_BOT = "🚀 Apply Config / Update Bots"
BUTTON_TEXT_STOP_TRADING_BOT = "🛑 Stop ALL Trading Bots"
BUTTON_TEXT_GET_LOGS = "📄 Recent Logs"
BUTTON_TEXT_CONFIGURE_BOTS = "⚙️ Configure/Edit Bots"
BUTTON_TEXT_EDIT_GLOBAL_DEFAULTS = "🛠️ Global Settings"
BUTTON_TEXT_TRAIN_MODELS = "👨‍🏫 Train / Update Models"
BUTTON_TEXT_CLEAR_CONFIG = "🗑️ Clear ALL Bot Configs"


# --- Helper to get chat_id ---
def get_chat_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.effective_chat:
        return update.effective_chat.id
    return 0

def load_config_from_file(context: ContextTypes.DEFAULT_TYPE):
    """Loads configurations from the JSON file at startup."""
    try:
        with open(ACTIVE_BOT_CONFIG_FILE, 'r', encoding='utf-8') as f:
            configs_from_file = json.load(f)
        
        user_configs = []
        for orchestrator_conf in configs_from_file:
            bot_config = orchestrator_conf.get("config", {})
            # Add symbol to config if it's missing there
            if 'symbol' not in bot_config:
                bot_config['symbol'] = orchestrator_conf.get('symbol')
            user_configs.append(bot_config)

        context.user_data['bot_configs_list'] = user_configs
        logger.info(f"Loaded {len(user_configs)} configurations from file {ACTIVE_BOT_CONFIG_FILE}")
    except (FileNotFoundError, json.JSONDecodeError):
        context.user_data['bot_configs_list'] = []
        logger.info("Configuration file not found or empty. Starting with a clean slate.")

# In file telegram_manager_2.py

# In file telegram_manager_2.py

def find_running_training_process() -> dict | None:
    """
    Scans processes on the server looking for an active training script.
    Returns a dictionary {'pid': ..., 'symbol': ...} or None.
    """
    try:
        # Looking for a unique signature - presence of '--train' in the command
        cmd_signature = "new_logic_2.py --train"
        for proc in psutil.process_iter(['pid', 'cmdline']):
            try:
                if proc.info['cmdline'] and len(proc.info['cmdline']) > 1:
                    cmdline = ' '.join(proc.info['cmdline'])
                    if cmd_signature in cmdline:
                        # Process found. Now extracting the symbol.
                        parts = proc.info['cmdline']
                        if '--symbol' in parts:
                            symbol_index = parts.index('--symbol') + 1
                            if symbol_index < len(parts):
                                symbol = parts[symbol_index]
                                pid = proc.info['pid']
                                logger.info(f"Found active training process for {symbol} with PID: {pid}.")
                                return {'pid': pid, 'symbol': symbol}
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception as e:
        logger.error(f"Error while scanning training processes: {e}")
    return None


def parse_training_summary(stdout_log: str) -> str:
    """
    Updated version for new_logic_2.py.
    Looks for 'VALIDATION RESULTS:' and formats the table.
    """
    try:
        # 1. Keyword from new_logic_2.py (function run_validation_only)
        # NOTE: logic in new_logic_2.py must output "VALIDATION RESULTS:" instead of Ukrainian text
        summary_start_keyword = "VALIDATION RESULTS:"
        
        table_start_index = stdout_log.rfind(summary_start_keyword)
        
        # Fallback for compatibility or other modes
        if table_start_index == -1:
             table_start_index = stdout_log.rfind("FINAL SUMMARY")

        if table_start_index != -1:
            # Look for the separator line that comes AFTER the table
            # In new_logic_2.py this is a line "="*50
            # We are looking for the closing separator
            
            # Step down a bit so we don't find the top separator
            search_start = table_start_index + len(summary_start_keyword)
            
            # Look for end of block (double line or end of output)
            end_separator = "=================================================="
            table_end_index = stdout_log.find(end_separator, search_start)

            if table_end_index != -1:
                # Capture a bit more context (capturing the top separator would be nice, but harder)
                # Take from the beginning of the found key to the separator
                # Look slightly higher to capture the top line
                block_start = stdout_log.rfind("=====", 0, table_start_index)
                if block_start == -1: block_start = table_start_index
                
                full_table_block = stdout_log[block_start : table_end_index + len(end_separator)]
                
                logger.info("Found and parsed final summary table.")
                safe_table_block = html.escape(full_table_block)
                return f"<pre>\n{safe_table_block}\n</pre>"

        logger.warning("Could not find full summary table.")
        
        # Check for successful training (without validation)
        # NOTE: new_logic_2.py must output these English phrases
        if "Training completed" in stdout_log and "Model saved" in stdout_log:
             return "✅ Model successfully trained and saved! (Mode --train-only, so no result table)."
        
        return "⚠️ Script finished but report not found. Check full log (/logs)."

    except Exception as e:
        logger.error(f"Report parsing error: {e}")
        return f"Report parsing error: {e}. Check full log."

async def set_bot_commands(application: Application):
    commands = [
        BotCommand("start", "🚀 Start manager and show main menu"),
        BotCommand("configure_bots", "⚙️ Configure/edit individual bot settings"),
        BotCommand("global_settings", "🛠️ Change global parameters for all bots"),
        BotCommand("apply_run", "▶️ Apply configuration and run/update bots"),
        BotCommand("stop_all", "⏹️ Stop all trading bots"),
        BotCommand("logs", "📊 Get recent logs"),
        BotCommand("clear_all_configs", "🗑️ Clear all individual bot configurations"),
        BotCommand("cancel", "❌ Cancel current configuration operation"),
    ]
    await application.bot.set_my_commands(commands)

async def display_main_menu(update_obj: Update | None,
                            context: ContextTypes.DEFAULT_TYPE,
                            message_text: str = None) -> int:
    chat_id = 0
    current_message_obj = None

    if isinstance(update_obj, Update) and update_obj.effective_message:
        current_message_obj = update_obj.effective_message
        chat_id = current_message_obj.chat_id
    elif hasattr(update_obj, 'message') and update_obj.message and hasattr(update_obj.message, 'chat_id'):
        current_message_obj = update_obj.message
        chat_id = current_message_obj.chat_id
    elif hasattr(update_obj, 'chat_id'):
        current_message_obj = update_obj
        chat_id = current_message_obj.chat_id
    
    if not chat_id and current_message_obj:
        chat_id = current_message_obj.chat_id
    
    if not chat_id:
        logger.error("Could not determine chat_id for display_main_menu")
        if isinstance(update_obj, Update) and update_obj.callback_query:
            try:
                await update_obj.callback_query.answer("Menu display error.")
                if update_obj.callback_query.message:
                    await update_obj.callback_query.message.reply_text("Could not determine chat. Try /start")
            except Exception as e_cb_ans:
                logger.error(f"Error answering callback_query in display_main_menu: {e_cb_ans}")
        return ConversationHandler.END

    context.bot_data.setdefault('authorized_chat_ids', set()).add(chat_id)

    # --- CHANGE HERE ---
    # Update keyboard, adding button for training
    keyboard = [
        [KeyboardButton(BUTTON_TEXT_CONFIGURE_BOTS), KeyboardButton(BUTTON_TEXT_TRAIN_MODELS)],
        [KeyboardButton(BUTTON_TEXT_APPLY_CONFIG_START_BOT), KeyboardButton(BUTTON_TEXT_STOP_TRADING_BOT)],
        [KeyboardButton(BUTTON_TEXT_GET_LOGS), KeyboardButton(BUTTON_TEXT_EDIT_GLOBAL_DEFAULTS)],
        [KeyboardButton(BUTTON_TEXT_CLEAR_CONFIG)],
    ]
    # --- END CHANGE ---

    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)
    text_to_send = message_text if message_text else 'Trading Bot Manager (Orchestrator).\nChoose an action:'
    
    try:
        if isinstance(update_obj, Update) and update_obj.callback_query:
            if update_obj.callback_query.message:
                await update_obj.callback_query.edit_message_reply_markup(reply_markup=None)
            await context.bot.send_message(chat_id=chat_id, text="Main Menu:", reply_markup=reply_markup)
        elif current_message_obj:
            await current_message_obj.reply_text(text_to_send, reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Error in display_main_menu: {e}")
    return ConversationHandler.END

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    chat_id = get_chat_id(update, context) # <--- FIXED
    logger.info(f"Command /start from chat_id {chat_id}")
    context.bot_data.setdefault('authorized_chat_ids', set()).add(chat_id)
    get_or_init_global_defaults(context.bot_data)
    load_config_from_file(context) # <--- ADDED
    await display_main_menu(update, context) # <--- CHANGED
    return ConversationHandler.END

async def ensure_authorized(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    chat_id = get_chat_id(update, context)
    if not chat_id:
        logger.warning("ensure_authorized: chat_id is 0. Authorization check failed.")
        return False
        
    if chat_id not in context.bot_data.get('authorized_chat_ids', set()):
        reply_message_obj = update.message if update.message else (update.callback_query.message if update.callback_query else None)
        if reply_message_obj:
            await reply_message_obj.reply_text("Please run /start to authorize first.")
        return False
    context.user_data.setdefault('bot_configs_list', [])
    context.user_data.setdefault('current_bot_config', {})
    return True

# --- Bot Configuration & Management Conversation (Individual Bots) ---
async def manage_bots_entry_point(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not await ensure_authorized(update, context): return ConversationHandler.END
    query = update.callback_query
    if query: await query.answer()
    await show_config_management_menu(update.effective_message, context, query=query)
    return MANAGE_BOT_CONFIG_MENU

async def show_config_management_menu(message_obj_for_reply, context: ContextTypes.DEFAULT_TYPE, query=None, message_prefix=""):
    configs = context.user_data.get('bot_configs_list', [])
    keyboard_inline = []
    text_for_message = message_prefix
    if not configs:
        text_for_message += "No individual bot configuration added yet.\n"
    else:
        text_for_message += "Current individual bot configurations:\n\n"
        for i, conf in enumerate(configs):
            symbol = conf.get('symbol', 'N/A')
            capital = conf.get('VIRTUAL_BOT_CAPITAL_USDT', 'N/A')
            text_for_message += (f"*{symbol}* (Capital: {capital} USDT)\n"
                                 f"  Retracement: `{conf.get('ENTRY_RETRACEMENT_PCT', 'N/A')}`, "
                                 f"TP: `{conf.get('TP_MULTIPLIER', 'N/A')}`, "
                                 f"SL: `{conf.get('SL_MULTIPLIER', 'N/A')}`\n\n")
            keyboard_inline.append([InlineKeyboardButton(f"❌ Delete {symbol}", callback_data=f"deletecfg_{symbol}_{i}")])
    keyboard_inline.append([InlineKeyboardButton("➕ Add New Bot", callback_data="cfg_add_bot")])
    keyboard_inline.append([InlineKeyboardButton("🏠 Return to Main Menu", callback_data="cfg_main_menu")])
    reply_markup_inline = InlineKeyboardMarkup(keyboard_inline)

    try:
        if query and query.message:
            await query.edit_message_text(text_for_message, reply_markup=reply_markup_inline, parse_mode="Markdown")
        elif message_obj_for_reply:
            await message_obj_for_reply.reply_text(text_for_message, reply_markup=reply_markup_inline, parse_mode="Markdown")
    except Exception as e:
        logger.warning(f"Error displaying config management menu: {e}")
        if message_obj_for_reply:
             await message_obj_for_reply.reply_text(text_for_message, reply_markup=reply_markup_inline, parse_mode="Markdown")


async def manage_config_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    data = query.data

    if data.startswith("deletecfg_"):
        try:
            parts = data.split('_')
            symbol_to_delete = parts[1]
            index_to_delete = int(parts[2])
            configs = context.user_data.get('bot_configs_list', [])
            if 0 <= index_to_delete < len(configs) and configs[index_to_delete]['symbol'] == symbol_to_delete:
                deleted_bot_info = configs.pop(index_to_delete)
                save_configs_to_file(context) # <--- ADDED SAVING
                logger.info(f"Deleted and saved configuration for {deleted_bot_info['symbol']} at index {index_to_delete}.")
                await show_config_management_menu(query.message, context, query=query, message_prefix=f"✅ Individual configuration for {symbol_to_delete} deleted.\n\n")
            else:
                await show_config_management_menu(query.message, context, query=query, message_prefix=f"⚠️ Error: could not delete.\n\n")
        except (IndexError, ValueError) as e:
            logger.error(f"Error processing callback 'deletecfg_': {data}, error: {e}")
            await show_config_management_menu(query.message, context, query=query, message_prefix="⚠️ Delete command format error.\n\n")
        return MANAGE_BOT_CONFIG_MENU
    elif data == "cfg_add_bot":
        context.user_data['current_bot_config'] = {} 
        if query.message: await query.edit_message_text("Enter trading pair symbol (e.g., ETHUSDT):")
        return ASK_SYMBOL 
    elif data == "cfg_main_menu":
        if query.message: await query.edit_message_text("Returning to main menu...")
        await display_main_menu(query, context)
        return ConversationHandler.END
    return MANAGE_BOT_CONFIG_MENU

async def ask_symbol_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    symbol = update.message.text.upper().strip()
    context.user_data['current_bot_config']['symbol'] = symbol
    await update.message.reply_text(f"Symbol: {symbol}. Now enter starting/trading capital (USDT, e.g., 100):")
    return ASK_CAPITAL

async def ask_capital_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        capital = float(update.message.text.strip())
        if capital <= 0:
            await update.message.reply_text("Capital must be a positive number. Try again:")
            return ASK_CAPITAL
            
        # Fill configuration
        context.user_data['current_bot_config']['VIRTUAL_BOT_CAPITAL_USDT'] = capital
        context.user_data['current_bot_config']['INITIAL_CAPITAL_PER_TRADE_USDT'] = capital
        # Set that bot uses ML, but without a specific threshold
        context.user_data['current_bot_config']['USE_ML_FILTER'] = True
        # --- ADDED NEW DEFAULT PARAMETER ---
        context.user_data['current_bot_config']['DIRECTION_CONFIDENCE_THRESHOLD'] = 0.60

        
        new_config = context.user_data['current_bot_config'].copy()
        
        # Add new config to list and save to file
        context.user_data.setdefault('bot_configs_list', []).append(new_config)
        save_configs_to_file(context)
        
        await update.message.reply_text(f"✅ Configuration for {new_config['symbol']} with capital {capital} USDT successfully added and saved.")
        
        # Clear temp data and return to config menu
        context.user_data['current_bot_config'] = {}
        await show_config_management_menu(update.message, context, query=None, message_prefix="")
        return MANAGE_BOT_CONFIG_MENU # Return to main settings menu

    except ValueError:
        await update.message.reply_text("Invalid capital format. Enter a number:")
        return ASK_CAPITAL
    
async def run_subprocess_and_notify(update: Update, context: ContextTypes.DEFAULT_TYPE, command: list, symbol: str):
    """Runs a subprocess, waits for completion, and notifies the user about the result, including logs."""
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        stdout_log = stdout.decode('utf-8', errors='ignore').strip()
        stderr_log = stderr.decode('utf-8', errors='ignore').strip()
        
        log_output_text = ""
        if stdout_log:
            log_output_text += f"\n\n--- Execution Log (STDOUT) ---\n`... (Log start skipped) ...\n{stdout_log[-1800:]}`"
        if stderr_log:
            log_output_text += f"\n\n--- Error Log (STDERR) ---\n`{stderr_log[:1500]}`"

        if process.returncode == 0:
            success_message = f"✅ Model for **{symbol}** successfully created!{log_output_text}"
            await context.bot.send_message(
                chat_id=update.effective_chat.id, 
                text=success_message, 
                parse_mode="Markdown"
            )
            # Automatically start/update orchestrator
            await apply_config_and_run_orchestrator(update, context, from_model_creation=True)

        else:
            error_message = f"❌ **ERROR** creating model for **{symbol}** (code: {process.returncode}).{log_output_text}"
            await context.bot.send_message(
                chat_id=update.effective_chat.id, 
                text=error_message, 
                parse_mode="Markdown"
            )

    except Exception as e:
        logging.error(f"Error running subprocess for {symbol}: {e}")
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"❌ Critical error attempting to start process for {symbol}. Details: `{e}`",
            parse_mode="Markdown"
        )
        
# In file telegram_manager_2.py
async def train_models_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    NEW VERSION: Entry point for training dialog.
    Shows mode selection (Train / Validate / 80-20).
    """
    global training_process_info
    if not await ensure_authorized(update, context): return ConversationHandler.END
    
    message_obj = update.effective_message
    
    keyboard = []
    text = ""
    
    if training_process_info:
        # If something is already training, show only that
        symbol = training_process_info.get('symbol', 'N/A')
        pid = training_process_info.get('pid', 'N/A')
        text = f"⚠️ **Training is already running for {symbol} (PID: {pid}).**\n"
        keyboard.append([InlineKeyboardButton(f"🛑 Stop training for {symbol}", callback_data=f"stop_train_{pid}")])
    else:
        # If nothing is training, show mode selection
        text = "👨‍🏫 **Training & Validation Menu**\n\nSelect mode:"
        keyboard.append([InlineKeyboardButton("1️⃣ Train 'Golden' Model (with dates)", callback_data="opt_train_only")])
        keyboard.append([InlineKeyboardButton("2️⃣ Run Validation (with dates)", callback_data="opt_validate_only")])
        keyboard.append([InlineKeyboardButton("3️⃣ [OLD] 80/20 Cycle (lookback 12m)", callback_data="opt_train_8020")])
    
    keyboard.append([InlineKeyboardButton("🏠 Return to Main Menu", callback_data="main_menu")])
    
    if update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")
    else:
        await message_obj.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")
        
    # Return state waiting for mode selection
    return SHOW_TRAINING_OPTIONS

def get_symbol_buttons(context: ContextTypes.DEFAULT_TYPE, callback_prefix: str) -> InlineKeyboardMarkup:
    """Helper function: creates buttons with symbols from config."""
    load_config_from_file(context) # Update list
    configs = context.user_data.get('bot_configs_list', [])
    keyboard = []
    if not configs:
        keyboard.append([InlineKeyboardButton("No symbols. Add in ⚙️ Settings", callback_data="main_menu")])
    else:
        for conf in configs:
            symbol = conf.get('symbol', 'N/A')
            keyboard.append([InlineKeyboardButton(f"{symbol}", callback_data=f"{callback_prefix}_{symbol}")])
    keyboard.append([InlineKeyboardButton("◀️ Back to mode selection", callback_data="train_menu_back")])
    return InlineKeyboardMarkup(keyboard)

async def training_options_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    STEP 1: Handles mode selection (Train / Validate / 80-20).
    """
    global training_process_info
    query = update.callback_query
    await query.answer()
    data = query.data
    context.user_data['training_job'] = {} # Clear data for new job

    if data == "main_menu":
        await query.edit_message_text("Returning to main menu...")
        new_update = Update(update.update_id, message=query.message)
        await display_main_menu(new_update, context)
        return ConversationHandler.END
    
    if data == "train_menu_back":
        # Return to initial menu (train_models_entry)
        new_update = Update(update.update_id, message=query.message)
        return await train_models_entry(new_update, context) # This returns SHOW_TRAINING_OPTIONS

    if data.startswith("stop_train_"):
        # Stop process logic (as before)
        try:
            pid_to_kill = int(data.split('_')[-1])
            if psutil.pid_exists(pid_to_kill):
                p = psutil.Process(pid_to_kill)
                p.terminate()
                await query.edit_message_text(f"✅ Stop command sent to process with PID {pid_to_kill}.")
            else:
                await query.edit_message_text("⚠️ Process no longer exists.")
        except Exception as e:
            await query.edit_message_text(f"⚠️ Stop error: {e}")
        finally:
            training_process_info.clear()
        
        await asyncio.sleep(1)
        new_update = Update(update.update_id, message=query.message)
        return await train_models_entry(new_update, context) # Restart menu

    # --- Routing to symbol selection ---
    if data == "opt_train_only":
        await query.edit_message_text("1️⃣ 'Golden' Model Training.\n\nSelect symbol for training:",
                                      reply_markup=get_symbol_buttons(context, "t_only_sym"))
        return TRAIN_ONLY_ASK_SYMBOL
        
    if data == "opt_validate_only":
        await query.edit_message_text("2️⃣ Model Validation.\n\nSelect symbol for validation:",
                                      reply_markup=get_symbol_buttons(context, "v_only_sym"))
        return VALIDATE_ONLY_ASK_SYMBOL

    if data == "opt_train_8020":
        await query.edit_message_text("3️⃣ Old Cycle (80/20).\n\nSelect symbol to launch:",
                                      reply_markup=get_symbol_buttons(context, "t_8020_sym"))
        return TRAIN_8020_ASK_SYMBOL
        
    return SHOW_TRAINING_OPTIONS

# --- Handlers for "Old Cycle 80/20" ---
async def train_8020_ask_symbol_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    STEP 2 (for 80/20): Got symbol, launch task.
    """
    query = update.callback_query
    await query.answer()
    
    if query.data == "train_menu_back":
        new_update = Update(update.update_id, message=query.message)
        return await train_models_entry(new_update, context) # Returns SHOW_TRAINING_OPTIONS

    symbol = query.data.split('_')[-1]
    await query.edit_message_text(f"🤖 Launching [Old Cycle 80/20] for **{symbol}**...", parse_mode="Markdown")

    python_executable = sys.executable
    # USING NEW FLAG --train-and-validate
    command = [python_executable, TRADING_BOT_SCRIPT, "--train-and-validate", "--symbol", symbol, "--no-progress-bar"]
    
    # Launch async task
    asyncio.create_task(launch_training_task(update, context, command, symbol))
    
    await display_main_menu(query, context, message_text=f"✅ Training task (80/20) for {symbol} launched in background.")
    return ConversationHandler.END

# --- Handlers for "Train Only" ---
async def train_only_ask_symbol_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """STEP 2 (for Train Only): Got symbol, ask start date."""
    query = update.callback_query
    await query.answer()

    if query.data == "train_menu_back":
        new_update = Update(update.update_id, message=query.message)
        return await train_models_entry(new_update, context)

    symbol = query.data.split('_')[-1]
    context.user_data['training_job'] = {'symbol': symbol, 'type': 'train_only'}
    
    await query.edit_message_text(f"Training for: *{symbol}*.\n\nEnter training **START** date.\nFormat: `YYYY-MM-DD` (e.g., `2025-01-01`)", parse_mode="Markdown")
    return TRAIN_ONLY_ASK_START

async def train_only_ask_start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """STEP 3 (for Train Only): Got start date, ask end date."""
    date_text = update.message.text.strip()
    # --- CHANGE: Updated Regex and message ---
    if not re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$', date_text):
        await update.message.reply_text("❌ Invalid format.\nEnter **START** time (e.g., `2025-01-01 00:00:00`):")
        return TRAIN_ONLY_ASK_START
        
    context.user_data['training_job']['start_date'] = date_text # Store full time
    
    await update.message.reply_text(f"Start Date (UTC): `{date_text}`.\n\nEnter training **END** time (UTC).\nFormat: `YYYY-MM-DD HH:MM:SS` (e.g., `2025-10-20 00:00:00`)", parse_mode="Markdown")
    # --- END CHANGE ---
    return TRAIN_ONLY_ASK_END

async def train_only_ask_end_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """STEP 4 (for Train Only): Got end date, launch task."""
    date_text = update.message.text.strip()
    # --- CHANGE: Updated Regex and message ---
    if not re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$', date_text):
        await update.message.reply_text("❌ Invalid format.\nEnter **END** time (e.g., `2025-10-20 00:00:00`):")
        return TRAIN_ONLY_ASK_END
        
    context.user_data['training_job']['end_date'] = date_text # Store full time
    # --- END CHANGE ---
    
    job_data = context.user_data['training_job']
    symbol = job_data['symbol']
    start_date = job_data['start_date']
    end_date = job_data['end_date']
    
    await update.message.reply_text(f"🤖 Launching ['Golden' Model Training] for **{symbol}**\n*Period (UTC):* `{start_date}` - `{end_date}`", parse_mode="Markdown")

    python_executable = sys.executable
    command = [
        python_executable, TRADING_BOT_SCRIPT, "--train-only",
        "--symbol", symbol,
        "--train-start", start_date,
        "--train-end", end_date,
        "--no-progress-bar"
    ]
    
    asyncio.create_task(launch_training_task(update, context, command, symbol))
    
    await display_main_menu(update, context, message_text=f"✅ Training task ({symbol}) launched in background.")
    context.user_data.pop('training_job', None)
    return ConversationHandler.END

# --- Handlers for "Validate Only" ---
async def validate_only_ask_symbol_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """STEP 2 (for Validate Only): Got symbol, ask start date."""
    query = update.callback_query
    await query.answer()
    
    if query.data == "train_menu_back":
        new_update = Update(update.update_id, message=query.message)
        return await train_models_entry(new_update, context)

    symbol = query.data.split('_')[-1]
    context.user_data['training_job'] = {'symbol': symbol, 'type': 'validate_only'}
    
    await query.edit_message_text(f"Validation for: *{symbol}*.\n\nEnter validation **START** date.\nFormat: `YYYY-MM-DD` (e.g., `2025-10-20`)", parse_mode="Markdown")
    return VALIDATE_ONLY_ASK_START

async def validate_only_ask_start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """STEP 3 (for Validate Only): Got start date, ask end date."""
    date_text = update.message.text.strip()
    # --- CHANGE: Updated Regex and message ---
    if not re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$', date_text):
        await update.message.reply_text("❌ Invalid format.\nEnter **START** time (e.g., `2025-10-20 00:00:00`):")
        return VALIDATE_ONLY_ASK_START
        
    context.user_data['training_job']['start_date'] = date_text # Store full time
    
    await update.message.reply_text(f"Start Date (UTC): `{date_text}`.\n\nEnter validation **END** time (UTC).\nFormat: `YYYY-MM-DD HH:MM:SS` (e.g., `2025-10-22 00:00:00`)", parse_mode="Markdown")
    # --- END CHANGE ---
    return VALIDATE_ONLY_ASK_END

async def validate_only_ask_end_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """STEP 4 (for Validate Only): Got end date, launch task."""
    date_text = update.message.text.strip()
    # --- CHANGE: Updated Regex and message ---
    if not re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$', date_text):
        await update.message.reply_text("❌ Invalid format.\nEnter **END** time (e.g., `2025-10-22 00:00:00`):")
        return VALIDATE_ONLY_ASK_END
        
    context.user_data['training_job']['end_date'] = date_text # Store full time
    # --- END CHANGE ---
    
    job_data = context.user_data['training_job']
    symbol = job_data['symbol']
    start_date = job_data['start_date']
    end_date = job_data['end_date']
    
    await update.message.reply_text(f"📊 Launching [Model Validation] for **{symbol}**\n*Period (UTC):* `{start_date}` - `{end_date}`", parse_mode="Markdown")

    python_executable = sys.executable
    command = [
        python_executable, TRADING_BOT_SCRIPT, "--validate-only",
        "--symbol", symbol,
        "--test-start", start_date,
        "--test-end", end_date,
        "--no-progress-bar"
    ]
    
    asyncio.create_task(launch_training_task(update, context, command, symbol))
    
    await display_main_menu(update, context, message_text=f"✅ Validation task ({symbol}) launched in background.")
    context.user_data.pop('training_job', None)
    return ConversationHandler.END
    
async def launch_training_task(update: Update, context: ContextTypes.DEFAULT_TYPE, command: list, symbol: str):
    """
    NEW UNIVERSAL FUNCTION:
    Launches any task (train, validate, 80/20),
    tracks PID and sends report.
    """
    global training_process_info
    chat_id = get_chat_id(update, context)
    
    await context.bot.send_message(
        chat_id=chat_id,
        text=f"👨‍🏫 Task for <b>{symbol}</b> started. Executing... (This may take minutes)",
        parse_mode="HTML"
    )

    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=1024 * 1024 * 10  # <--- ADD THIS LINE (10 MB buffer)
        )
        
        # Save PID
        training_process_info = {'pid': process.pid, 'symbol': symbol}
        logger.info(f"Process ({' '.join(command)}) started with PID: {process.pid}.")

        stdout_lines = []
        stderr_lines = []

        async def read_stream(stream, container):
            while True:
                line = await stream.readline()
                if not line: break
                container.append(line.decode('utf-8', errors='ignore'))

        await asyncio.gather(
            read_stream(process.stdout, stdout_lines),
            read_stream(process.stderr, stderr_lines)
        )

        await process.wait() # Wait for completion
        logger.info(f"Process for {symbol} (PID: {process.pid}) finished with code {process.returncode}.")

        stdout_log = "".join(stdout_lines)
        stderr_log = "".join(stderr_lines)

        if process.returncode == 0:
            
            # --- FIX START ---
            if "--train-only" in command:
                # No report needed for --train-only!
                task_type = "'Golden' Model Training"
                success_message = f"✅ {task_type} for <b>{symbol}</b> completed successfully!\n\n(Models and scaler saved. Validation skipped.)"
            else:
                # For --validate-only and --train-and-validate, parse report
                summary = parse_training_summary(stdout_log)
                task_type = "Validation" if "--validate-only" in command else "Cycle (80/20)"
                success_message = f"✅ {task_type} for <b>{symbol}</b> completed successfully!\n\n{summary}"
            # --- FIX END ---

            await context.bot.send_message(chat_id=chat_id, text=success_message, parse_mode="HTML")
        else:
            # Error
            safe_stderr = html.escape(stderr_log[-1500:])
            safe_stdout = html.escape(stdout_log[-1500:])
            full_log = f"--- Error Log ---\n<pre>{safe_stderr}</pre>\n\n--- Standard Output ---\n<pre>{safe_stdout}</pre>"
            error_message = f"❌ <b>ERROR</b> during task for <b>{symbol}</b>.\n\n{full_log}"
            await context.bot.send_message(chat_id=chat_id, text=error_message, parse_mode="HTML")
            
    except Exception as e:
        safe_exception_text = html.escape(str(e))
        error_message = f"❌ Critical error running subprocess for {symbol}: <pre>{safe_exception_text}</pre>"
        await context.bot.send_message(chat_id=chat_id, text=error_message, parse_mode="HTML")
    finally:
        # Clear PID when task finished (success or fail)
        training_process_info.clear()
        logger.info(f"Process info (PID: {process.pid if process else 'N/A'}) cleared.")

async def run_subprocess_and_get_summary(command: list) -> Tuple[bool, str]:
    """Runs a subprocess, returns status and a concise summary, tracking the process."""
    global training_process_info
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        # Save info about the running process
        symbol = command[command.index("--symbol") + 1]
        training_process_info = {'process': process, 'symbol': symbol}
        logger.info(f"Training process for {symbol} started with PID: {process.pid}")

        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            summary = parse_training_summary(stdout.decode('utf-8', errors='ignore'))
            return True, summary
        else:
            error_log = stderr.decode('utf-8', errors='ignore').strip()
            # Add stdout as well, it might contain useful logs
            stdout_log = stdout.decode('utf-8', errors='ignore').strip()
            full_log = f"--- Error Log ---\n`{error_log[-1500:]}`\n\n--- Standard Output ---\n`{stdout_log[-1500:]}`"
            return False, full_log
    except Exception as e:
        return False, f"Critical subprocess launch error: `{e}`"
    finally:
        # Guarantee cleanup of process info after completion
        if process:
             logger.info(f"Training process for {training_process_info.get('symbol')} finished with code {process.returncode}.")
        training_process_info.clear()

# In file telegram_manager_2.py

def get_or_init_global_defaults(bot_data: dict) -> dict:
    """
    Initializes or UPDATES global parameters.
    Always synchronizes saved parameters with the actual code.
    """
    # 1. Define "reference" set of parameters directly from code.
    hardcoded_defaults = {
        "interval": "5m",
        "SELECTED_TIMEFRAME_HIGHER": "2h",
        "ATR_PERIOD": 14,
        "LOG_LEVEL": "INFO",
        "TELEGRAM_LOG_LEVEL": "INFO",
        "LIMIT_MAIN_TF": 1500,
        "LIMIT_HIGHER_TF": 1500,
        "RESIDUAL_SMOOTH": 10,
        "TARGET_PERCENT_PARAM": 20.0,
        "BOT_VIRTUAL_CAPITAL_STOP_LOSS_PERCENT": "0.60",
        "MIN_DISTANCE_FROM_ENTRY_TICKS": "5",
        "LIQUIDATION_PRICE_BUFFER_TICKS": "10",
        "MAX_RISK_PERCENT_OF_PER_TRADE_CAPITAL": "0.7",
        "MAX_WEBSOCKET_RESTART_ATTEMPTS": 3,
        # --- CHANGE HERE: Added new global parameter ---
        # "ENSEMBLE_VOTE_THRESHOLD": 5 
        "STEP_COEFF": 0.5
    }
    
    # 2. Check if anything is in bot "memory" (persistence).
    if 'global_bot_defaults' in bot_data:
        saved_defaults = bot_data['global_bot_defaults']
        
        # 3. Synchronization:
        # Remove from saved settings those that are no longer in code.
        keys_to_remove = [key for key in saved_defaults if key not in hardcoded_defaults]
        if keys_to_remove:
            logger.info(f"Removing obsolete global parameters: {keys_to_remove}")
            for key in keys_to_remove:
                del saved_defaults[key]
        
        # Add new parameters to saved settings if they appeared in code.
        keys_to_add = {key: value for key, value in hardcoded_defaults.items() if key not in saved_defaults}
        if keys_to_add:
            logger.info(f"Adding new global parameters: {list(keys_to_add.keys())}")
            saved_defaults.update(keys_to_add)
            
        bot_data['global_bot_defaults'] = saved_defaults
        return saved_defaults

    # 4. If memory is empty (first run), just use reference settings.
    else:
        logger.info(f"Global default parameters initialized (first run).")
        bot_data['global_bot_defaults'] = hardcoded_defaults
        return hardcoded_defaults


async def edit_global_defaults_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not await ensure_authorized(update, context): return ConversationHandler.END
    get_or_init_global_defaults(context.bot_data)
    query = update.callback_query
    message_to_use = update.effective_message
    if query: 
        await query.answer()
        message_to_use = query.message
        
    await show_global_params_menu_telegram(message_to_use, context, query=query)
    return SHOW_GLOBAL_PARAMS_MENU

async def show_global_params_menu_telegram(message_obj_for_reply, context: ContextTypes.DEFAULT_TYPE, query=None, prefix_text=""):
    global_defaults = get_or_init_global_defaults(context.bot_data)
    keyboard_inline = []
    text_for_message = prefix_text + "Current global parameters for all bots:\n(Click a parameter to edit)\n\n"
    
    param_types = {
        "ATR_PERIOD": int, "LIMIT_MAIN_TF": int, "LIMIT_HIGHER_TF": int, 
        "RESIDUAL_SMOOTH": int, "MAX_WEBSOCKET_RESTART_ATTEMPTS": int,
        "PROBABILITY_PARAM": float, "TARGET_PERCENT_PARAM": float,
        "MIN_TP_MULTIPLIER": float, "MAX_TP_MULTIPLIER": float,
        "MIN_SL_MULTIPLIER": float, "MAX_SL_MULTIPLIER": float,
        "STEP_COEFF": float # <--- ADD THIS
    }

    for key, value in global_defaults.items():
        expected_type_name = param_types.get(key, str).__name__
        text_for_message += f"*{key}* (`{expected_type_name}`): `{value}`\n"
        keyboard_inline.append([InlineKeyboardButton(f"Edit {key}", callback_data=f"editglobal_{key}")])
    
    keyboard_inline.append([InlineKeyboardButton("🏠 Return to Main Menu", callback_data="editglobal_main_menu")])
    reply_markup = InlineKeyboardMarkup(keyboard_inline)

    try:
        if query and query.message:
            await query.edit_message_text(text_for_message, reply_markup=reply_markup, parse_mode="Markdown")
        elif message_obj_for_reply:
            await message_obj_for_reply.reply_text(text_for_message, reply_markup=reply_markup, parse_mode="Markdown")
    except Exception as e:
        logger.warning(f"Error displaying global params menu: {e}")
        if message_obj_for_reply:
             await message_obj_for_reply.reply_text(text_for_message, reply_markup=reply_markup, parse_mode="Markdown")


async def global_params_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "editglobal_main_menu":
        if query.message: await query.edit_message_text("Returning to main menu...")
        await display_main_menu(query, context) 
        return ConversationHandler.END
    
    if data.startswith("editglobal_"):
        param_name_to_edit = data.split("editglobal_")[1]
        context.user_data['param_to_edit_global'] = param_name_to_edit
        current_value = get_or_init_global_defaults(context.bot_data).get(param_name_to_edit, "N/A")
        if query.message: await query.edit_message_text(f"Enter new value for global parameter *{param_name_to_edit}* (current: `{current_value}`):", parse_mode="Markdown")
        return AWAIT_GLOBAL_PARAM_VALUE_INPUT
    
    return SHOW_GLOBAL_PARAMS_MENU

# In file telegram_manager_2.py

async def received_global_param_value(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    param_name = context.user_data.pop('param_to_edit_global', None)
    new_value_str = update.message.text.strip()

    if not param_name:
        await update.message.reply_text("Error: could not determine parameter. Returning to menu.")
        await show_global_params_menu_telegram(update.message, context)
        return SHOW_GLOBAL_PARAMS_MENU

    global_defaults = get_or_init_global_defaults(context.bot_data)
    # Updated, cleaned type dict
    param_types = {
        "ATR_PERIOD": int,
        "LIMIT_MAIN_TF": int,
        "LIMIT_HIGHER_TF": int, 
        "RESIDUAL_SMOOTH": int,
        "MAX_WEBSOCKET_RESTART_ATTEMPTS": int,
        "TARGET_PERCENT_PARAM": float,
        "STEP_COEFF": float # <--- ADD THIS
    }
    target_type = param_types.get(param_name)
    original_value = global_defaults.get(param_name)
    converted_value = None

    try:
        if target_type == int: converted_value = int(new_value_str)
        elif target_type == float: converted_value = float(new_value_str)
        else: 
            converted_value = new_value_str
            if param_name in ['LOG_LEVEL', 'TELEGRAM_LOG_LEVEL'] and converted_value.upper() not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
                raise ValueError("Invalid value for log level.")
            if param_name in ['BOT_VIRTUAL_CAPITAL_STOP_LOSS_PERCENT', 
                              'MIN_DISTANCE_FROM_ENTRY_TICKS', 
                              'LIQUIDATION_PRICE_BUFFER_TICKS',
                              'MAX_RISK_PERCENT_OF_PER_TRADE_CAPITAL']:
                float(converted_value) 
    except ValueError as e_val:
        await update.message.reply_text(f"Invalid format for *{param_name}*: {e_val}. Expected type `{target_type.__name__ if target_type else type(original_value).__name__ if original_value is not None else 'string'}`. Try again.", parse_mode="Markdown")
        await show_global_params_menu_telegram(update.message, context, query=None, prefix_text=f"Input error for *{param_name}*.\n\n")
        return SHOW_GLOBAL_PARAMS_MENU

    global_defaults[param_name] = converted_value
    context.bot_data['global_bot_defaults'] = global_defaults
    logger.info(f"Global parameter {param_name} updated to '{converted_value}'")
    
    await show_global_params_menu_telegram(update.message, context, query=None, prefix_text=f"✅ Global parameter *{param_name}* updated to `{converted_value}`.\n\n")
    return SHOW_GLOBAL_PARAMS_MENU

async def cancel_global_defaults_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not await ensure_authorized(update, context): return ConversationHandler.END
    context.user_data.pop('param_to_edit_global', None) 
    if update.message:
        await update.message.reply_text("Global parameters editing cancelled.")
    await display_main_menu(update, context)
    return ConversationHandler.END

async def cancel_conversation_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not await ensure_authorized(update, context): return ConversationHandler.END
    context.user_data.pop('current_bot_config', None)
    context.user_data.pop('param_to_edit_global', None)
    if update.message:
        await update.message.reply_text("Operation cancelled. Returning to main menu.")
    elif update.callback_query:
        await update.callback_query.answer("Operation cancelled.")
        if update.callback_query.message:
             try: await update.callback_query.edit_message_reply_markup(reply_markup=None)
             except Exception: pass
    await display_main_menu(update, context)
    return ConversationHandler.END

# New code
async def clear_all_configs_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await ensure_authorized(update, context): return
    context.user_data['bot_configs_list'] = []
    context.user_data['current_bot_config'] = {}
    save_configs_to_file(context) # <--- ADDED SAVING
    logger.info(f"Individual bot configurations cleared and saved for chat_id {get_chat_id(update, context)}")
    await update.message.reply_text("✅ All individual bot configurations deleted.")

def save_configs_to_file(context: ContextTypes.DEFAULT_TYPE):
    """Collects configurations and saves them to a JSON file (FIXED VERSION)."""
    try:
        # Get updated global settings directly from memory
        global_defaults = context.bot_data.get('global_bot_defaults', {})
        if not global_defaults:
            logger.error("SAVE ERROR: Could not retrieve global settings from memory (bot_data).")
            return

        user_configs = context.user_data.get('bot_configs_list', [])
        
        final_configs = []
        for user_conf in user_configs:
            # Create full config for each bot: first globals, then individual
            full_config = global_defaults.copy()
            full_config.update(user_conf)
            final_configs.append({
                "symbol": user_conf.get("symbol"),
                "config": full_config
            })

        # DIAGNOSTIC MESSAGE
        tf_to_save = final_configs[0]['config'].get('SELECTED_TIMEFRAME_HIGHER', 'NOT FOUND') if final_configs else 'LIST EMPTY'
        logger.info(f"!!! BEFORE FILE WRITE: Saving timeframe = '{tf_to_save}'")

        with open(ACTIVE_BOT_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(final_configs, f, indent=4, ensure_ascii=False)
        
        logger.info(f"✅ Configurations ({len(final_configs)} bot(s)) successfully saved to {ACTIVE_BOT_CONFIG_FILE}")

    except Exception as e:
        logger.critical(f"CRITICAL ERROR in save_configs_to_file: {e}", exc_info=True)

        
# --- MODIFIED LAUNCH FUNCTION WITH RETURN OF OLD LOGIC ---
async def apply_config_and_run_orchestrator(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global trading_bot_process
    if not await ensure_authorized(update, context): return

    save_configs_to_file(context) # Save latest changes just in case
    num_bots = len(context.user_data.get('bot_configs_list', []))
    await update.effective_message.reply_text(f"💾 Configurations ({num_bots} bot(s)) saved.")

    try:
        if trading_bot_process and trading_bot_process.poll() is None:
            with open(RELOAD_CONFIG_FLAG_MANAGER, 'w') as f: f.write('reload')
            await update.effective_message.reply_text("✅ Configuration update command sent to running orchestrator.")
        else:
            if num_bots == 0:
                 await update.effective_message.reply_text("⚠️ Bot list is empty. Orchestrator will not be started.")
                 return

            python_executable = sys.executable
            bot_env = os.environ.copy()
            bot_env["YOUR_TELEGRAM_CHAT_ID"] = str(get_chat_id(update, context)) # <--- FIXED


            trading_bot_process = subprocess.Popen(
                [python_executable, TRADING_BOT_SCRIPT, '--run-orchestrator', '--config', ACTIVE_BOT_CONFIG_FILE], env=bot_env, cwd=SCRIPT_DIR
            )
            await update.effective_message.reply_text(f"🚀 Orchestrator ({num_bots} instances) starting (PID: {trading_bot_process.pid})...")
            
            await asyncio.sleep(5)
            if trading_bot_process.poll() is None:
                await update.effective_message.reply_text(f"✅ Orchestrator successfully started!")
            else:
                await update.effective_message.reply_text(f"⚠️ Orchestrator launch error (code: {trading_bot_process.returncode}).")
                trading_bot_process = None
    except Exception as e:
        await update.effective_message.reply_text(f"❌ Launch/Update error: {e}")

# --- RESTORED OLD STOP FUNCTION ---
async def stop_trading_bot_orchestrator_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global trading_bot_process
    if not await ensure_authorized(update, context): return
    if not update.message: return 

    if not trading_bot_process or trading_bot_process.poll() is not None:
        await update.message.reply_text("Trading Bot Orchestrator is not running (or already stopped).")
        if trading_bot_process:
             logger.info(f"Stop command for already stopped orchestrator (PID: {trading_bot_process.pid}, Code: {trading_bot_process.returncode})")
             trading_bot_process = None
        else:
             logger.info("Stop command when process is None.")
        return
    try:
        with open(GLOBAL_STOP_FLAG_FILE, 'w', encoding='utf-8') as f:
            f.write('stop')
        logger.info(f"Global stop flag file {GLOBAL_STOP_FLAG_FILE} created to stop orchestrator.")
        await update.message.reply_text("🛑 Sending STOP command to ALL trading bots via orchestrator...")
        timeout_seconds = 45
        logger.info(f"Waiting for orchestrator stop up to {timeout_seconds} seconds...")
        try:
            trading_bot_process.wait(timeout=timeout_seconds)
            await update.message.reply_text(f"✅ Trading Bot Orchestrator successfully stopped (PID: {trading_bot_process.pid}, code: {trading_bot_process.returncode}).")
            logger.info(f"Orchestrator successfully stopped (PID: {trading_bot_process.pid}, code: {trading_bot_process.returncode}).")
        except subprocess.TimeoutExpired:
            logger.warning(f"Orchestrator (PID: {trading_bot_process.pid}) did not stop in time. Forcing termination...")
            trading_bot_process.terminate() 
            await asyncio.sleep(5) 
            if trading_bot_process.poll() is None:
                logger.warning(f"Orchestrator (PID: {trading_bot_process.pid}) did not stop after SIGTERM. Sending SIGKILL...")
                trading_bot_process.kill()
                await asyncio.sleep(1)
            await update.message.reply_text("⚠️ Trading Bot Orchestrator forcibly stopped.")
            logger.info(f"Orchestrator (PID: {trading_bot_process.pid}) forcibly stopped.")
        except Exception as e_wait:
             logger.error(f"Error waiting for stop: {e_wait}", exc_info=True)
             await update.message.reply_text(f"❌ Error waiting for stop: {e_wait}")
    except Exception as e:
        logger.error(f"Error in stop_trading_bot_orchestrator_handler: {e}", exc_info=True)
        await update.message.reply_text(f"❌ General orchestrator stop error: {e}")
    finally:
        trading_bot_process = None 
        if os.path.exists(GLOBAL_STOP_FLAG_FILE):
             try: os.remove(GLOBAL_STOP_FLAG_FILE)
             except Exception: pass

async def get_logs_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await ensure_authorized(update, context): return
    if not update.message: return

    configured_bots = context.user_data.get('bot_configs_list', [])
    if not configured_bots:
        await update.message.reply_text("No configured bots to show logs.")
        return
    buttons = [[InlineKeyboardButton(f"Logs for {b.get('symbol','N/A')}", callback_data=f"лог_{b.get('symbol','N/A')}")] for b in configured_bots]
    buttons.append([InlineKeyboardButton("❌ Cancel", callback_data="лог_скасувати")])
    reply_markup = InlineKeyboardMarkup(buttons)
    await update.message.reply_text("Select symbol to retrieve logs:", reply_markup=reply_markup)

async def log_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "лог_скасувати":
        await query.edit_message_text("Log request cancelled.")
        return

    if data.startswith("лог_"):
        symbol = data.split("_", 1)[1]
        log_file_name = f"trading_bot_{symbol}_sessions.log"
        log_file_path = os.path.join(SCRIPT_DIR, log_file_name)
        await query.edit_message_text(f"⏳ Retrieving recent logs for {symbol}...")
        
        chat_id = query.message.chat_id

        try:
            lines_to_check = tail(log_file_path, 20000)
            if not lines_to_check:
                 if os.path.exists(log_file_path):
                     await context.bot.send_message(chat_id=chat_id, text=f"Could not read logs for {symbol}. File might be empty.")
                 else:
                     await context.bot.send_message(chat_id=chat_id, text=f"Log file {log_file_name} not found.")
                 return

            keywords = ["completed", "stopped", "stop", "error", "closed"]
            relevant_lines = []
            for line in reversed(lines_to_check):
                if any(keyword in line.lower() for keyword in keywords):
                    relevant_lines.append(line)
                if len(relevant_lines) >= 30:
                    break
            relevant_lines.reverse()

            if relevant_lines:
                log_message = f"📄 Recent {len(relevant_lines)} important logs for {symbol}:\n" + "".join(relevant_lines)
                if len(log_message) > 4000:
                    for i in range(0, len(log_message), 4000):
                        await context.bot.send_message(chat_id=chat_id, text=log_message[i:i+4000])
                else:
                    await context.bot.send_message(chat_id=chat_id, text=log_message)
            else:
                await context.bot.send_message(chat_id=chat_id, text=f"No important logs found for {symbol} (checked last {len(lines_to_check)} lines).")
                
        except Exception as e:
            logger.error(f"Error retrieving logs for {symbol}: {e}", exc_info=True)
            await context.bot.send_message(chat_id=chat_id, text=f"❌ Error retrieving logs: {e}")

# In file telegram_manager_2.py

async def handle_text_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles main menu button clicks which are one-time actions.
    """
    if not await ensure_authorized(update, context): return
    if not update.message: return 
    
    text = update.message.text
    
    if text == BUTTON_TEXT_APPLY_CONFIG_START_BOT:
        await apply_config_and_run_orchestrator(update, context)
    elif text == BUTTON_TEXT_STOP_TRADING_BOT:
        await stop_trading_bot_orchestrator_handler(update, context)
    elif text == BUTTON_TEXT_GET_LOGS:
        await get_logs_handler(update, context)
    elif text == BUTTON_TEXT_CLEAR_CONFIG:
        await clear_all_configs_command_handler(update, context)
    elif text == BUTTON_TEXT_TRAIN_MODELS:
        # This button triggers a dialog start
        await train_models_entry(update, context)
    elif text == BUTTON_TEXT_CONFIGURE_BOTS:
        # This button starts a dialog, but if regex didn't catch it, we start here
        await manage_bots_entry_point(update, context)
    elif text == BUTTON_TEXT_EDIT_GLOBAL_DEFAULTS:
        # This button starts a dialog, but if regex didn't catch it, we start here
        await edit_global_defaults_entry(update, context)
    else:
        logger.info(f"Received unrecognized text '{text}'.")

async def post_init(application: Application):
    await set_bot_commands(application)
    logger.info("Bot commands set.")
    application.bot_data.setdefault('authorized_chat_ids', set())
    get_or_init_global_defaults(application.bot_data) 

def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        logger.critical("TELEGRAM_BOT_TOKEN not found!")
        return

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).build()

    # Conversation for BOT CONFIGURATION
    bot_individual_config_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("configure_bots", manage_bots_entry_point),
                      MessageHandler(filters.Regex(f"^{BUTTON_TEXT_CONFIGURE_BOTS}$"), manage_bots_entry_point)],
        states={
            MANAGE_BOT_CONFIG_MENU: [CallbackQueryHandler(manage_config_callback_handler)],
            ASK_SYMBOL: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_symbol_handler)],
            ASK_CAPITAL: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_capital_handler)],
             
        },
        # fallbacks=[
        #     MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_button), # <--- ADD THIS LINE
        #     CommandHandler("cancel", cancel_conversation_command), 
        #     CommandHandler("start", start_command)
        # ],
        fallbacks=[
            CommandHandler("cancel", cancel_conversation_command), 
            CommandHandler("start", start_command)
        ],
        name="bot_individual_config_conversation", per_user=True, per_chat=True
    )
    application.add_handler(bot_individual_config_conv_handler)

    # Conversation for GLOBAL SETTINGS
    global_defaults_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("global_settings", edit_global_defaults_entry),
                      MessageHandler(filters.Regex(f"^{BUTTON_TEXT_EDIT_GLOBAL_DEFAULTS}$"), edit_global_defaults_entry)],
        states={
            SHOW_GLOBAL_PARAMS_MENU: [CallbackQueryHandler(global_params_callback_handler)],
            AWAIT_GLOBAL_PARAM_VALUE_INPUT: [MessageHandler(filters.TEXT & ~filters.COMMAND, received_global_param_value)],
        },
        # fallbacks=[
        #     MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_button), # <--- ADD THIS LINE
        #     CommandHandler("cancel", cancel_global_defaults_conversation),
        #     CommandHandler("start", start_command)
        # ],
        fallbacks=[
            CommandHandler("cancel", cancel_global_defaults_conversation),
            CommandHandler("start", start_command)
        ],
        name="global_defaults_conversation", per_user=True, per_chat=True
    )
    application.add_handler(global_defaults_conv_handler)
# --- NEW DIALOG FOR TRAINING AND VALIDATION ---
    training_conv_handler = ConversationHandler(
        entry_points=[MessageHandler(filters.Regex(f"^{BUTTON_TEXT_TRAIN_MODELS}$"), train_models_entry)],
        states={
            # Step 1: Mode Selection
            SHOW_TRAINING_OPTIONS: [CallbackQueryHandler(training_options_callback)],
            
            # Branch 1: Train Only
            TRAIN_ONLY_ASK_SYMBOL: [CallbackQueryHandler(train_only_ask_symbol_callback, pattern="^t_only_sym_")],
            TRAIN_ONLY_ASK_START: [MessageHandler(filters.TEXT & ~filters.COMMAND, train_only_ask_start_handler)],
            TRAIN_ONLY_ASK_END: [MessageHandler(filters.TEXT & ~filters.COMMAND, train_only_ask_end_handler)],

            # Branch 2: Validate Only
            VALIDATE_ONLY_ASK_SYMBOL: [CallbackQueryHandler(validate_only_ask_symbol_callback, pattern="^v_only_sym_")],
            VALIDATE_ONLY_ASK_START: [MessageHandler(filters.TEXT & ~filters.COMMAND, validate_only_ask_start_handler)],
            VALIDATE_ONLY_ASK_END: [MessageHandler(filters.TEXT & ~filters.COMMAND, validate_only_ask_end_handler)],

            # Branch 3: Train 80/20 (Old logic)
            TRAIN_8020_ASK_SYMBOL: [CallbackQueryHandler(train_8020_ask_symbol_callback, pattern="^t_8020_sym_")],
        },
        fallbacks=[
            CallbackQueryHandler(training_options_callback, pattern="^train_menu_back$"), # Back button
            CommandHandler("cancel", cancel_conversation_command), 
            CommandHandler("start", start_command)
        ],
        name="training_conversation",
        per_user=True, per_chat=True
    )
    application.add_handler(training_conv_handler)
    # --- END NEW DIALOG ---
    
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("clear_all_configs", clear_all_configs_command_handler))
    application.add_handler(CommandHandler("apply_run", apply_config_and_run_orchestrator))
    application.add_handler(CommandHandler("stop_all", stop_trading_bot_orchestrator_handler))
    application.add_handler(CommandHandler("logs", get_logs_handler))
    application.add_handler(CallbackQueryHandler(log_callback_handler, pattern="^лог_"))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_button))

    find_and_set_existing_process() # <--- ADD THIS LINE
    
    logger.info("Telegram Manager (Orchestrator) started...")
    try:
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    except Exception as e:
        logger.critical(f"Critical error in Telegram Manager: {e}", exc_info=True)
    finally:
        logger.info("Telegram Manager stopped.")
        # Important to clear flags on stop
        for flag_file in [GLOBAL_STOP_FLAG_FILE, RELOAD_CONFIG_FLAG_MANAGER]:
            if os.path.exists(flag_file):
                try:
                    os.remove(flag_file)
                except OSError:
                    pass

if __name__ == '__main__':
    main()