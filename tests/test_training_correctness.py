"""Source-level regression tests for training and evaluation correctness.

The active test environment intentionally does not import the project's PyTorch
dependencies. These tests inspect syntax trees rather than importing modules.
"""

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def read_source(path):
    return (ROOT / path).read_text(encoding="utf-8")


def parse_module(path):
    return ast.parse(read_source(path), filename=str(ROOT / path))


def lookup_class(module, path, name):
    matches = [
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == name
    ]
    assert len(matches) == 1, f"{path}: expected exactly one class named {name}"
    return matches[0]


def lookup_function(container, path, name):
    matches = [
        node
        for node in container.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    assert len(matches) == 1, f"{path}: expected exactly one function/method named {name}"
    return matches[0]


def function_parameters(function_node):
    arguments = function_node.args
    return [
        argument.arg
        for argument in (
            [*arguments.posonlyargs, *arguments.args]
            + ([arguments.vararg] if arguments.vararg else [])
            + arguments.kwonlyargs
            + ([arguments.kwarg] if arguments.kwarg else [])
        )
        if argument is not None
    ]


def assignment_targets(assignment):
    if isinstance(assignment, ast.Assign):
        targets = assignment.targets
    elif isinstance(assignment, ast.AnnAssign):
        targets = [assignment.target]
    else:
        return []
    return [target.id for target in targets if isinstance(target, ast.Name)]


def assignment_map(scope):
    result = {}
    for node in ast.walk(scope):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            for target in assignment_targets(node):
                result.setdefault(target, []).append(node.value)
    return result


def expression_names(expression):
    return {node.id for node in ast.walk(expression) if isinstance(node, ast.Name)}


def depends_on(expression, source_names, assignments, seen=None):
    """Whether an expression's assignment dataflow reaches a source name."""
    seen = set() if seen is None else seen
    if isinstance(expression, ast.Name):
        if expression.id in source_names:
            return True
        if expression.id in seen:
            return False
        return any(
            depends_on(value, source_names, assignments, seen | {expression.id})
            for value in assignments.get(expression.id, [])
        )
    return any(
        depends_on(child, source_names, assignments, seen)
        for child in ast.iter_child_nodes(expression)
    )


def is_named_call(node, name):
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name


def is_attribute_call(node, object_name, method_name):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == method_name
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == object_name
    )


def bound_argument(call, function_node, parameter_name):
    """Return the AST expression bound to a formal parameter, if any."""
    for keyword in call.keywords:
        if keyword.arg == parameter_name:
            return keyword.value

    parameters = function_parameters(function_node)
    if parameters and parameters[0] == "self":
        parameters = parameters[1:]
    if parameter_name not in parameters:
        return None
    position = parameters.index(parameter_name)
    return call.args[position] if position < len(call.args) else None


def position_parameter(function_node, path, symbol):
    parameters = function_parameters(function_node)
    matches = [name for name in parameters if name == "position_ids"]
    if not matches:
        matches = [name for name in parameters if "position" in name]
    assert matches, f"{path}: {symbol} must accept a position-id parameter"
    return matches[0]


def is_lm_checkpoint_call(node):
    return is_named_call(node, "lm_checkpoint") and any(
        keyword.arg == "model" for keyword in node.keywords
    )


def flattened_and_conjuncts(expression):
    if isinstance(expression, ast.BoolOp) and isinstance(expression.op, ast.And):
        return [
            conjunct
            for value in expression.values
            for conjunct in flattened_and_conjuncts(value)
        ]
    return [expression]


def has_positive_and_conjunct(expression, name):
    return isinstance(expression, ast.BoolOp) and isinstance(expression.op, ast.And) and any(
        isinstance(conjunct, ast.Name) and conjunct.id == name
        for conjunct in flattened_and_conjuncts(expression)
    )


def parent_map(root):
    return {
        child: node
        for node in ast.walk(root)
        for child in ast.iter_child_nodes(node)
    }


def enclosing_nodes(node, parents):
    current = parents.get(node)
    while current is not None:
        yield current
        current = parents.get(current)


def is_sample_subscript(node, field):
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "sample"
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == field
    )


def is_sample_get_call(node, field):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "sample"
        and node.func.attr == "get"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == field
    )


def is_sample_membership_check(node, field):
    return (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], (ast.In, ast.NotIn))
        and isinstance(node.left, ast.Constant)
        and node.left.value == field
        and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Name)
        and node.comparators[0].id == "sample"
    )


def is_raw_prompt_check(expression):
    return any(
        is_sample_get_call(node, "prompt")
        or is_sample_subscript(node, "prompt")
        or is_sample_membership_check(node, "prompt")
        for node in ast.walk(expression)
    )


def dict_value(dictionary, key_name):
    for key, value in zip(dictionary.keys, dictionary.values):
        if isinstance(key, ast.Constant) and key.value == key_name:
            return value
    return None


def raw_prompt_branch(getitem, path):
    branches = [
        node
        for node in ast.walk(getitem)
        if isinstance(node, ast.If)
        and is_raw_prompt_check(node.test)
    ]
    assert len(branches) == 1, (
        f"{path}: RLAIFDataset.__getitem__ must have one raw-prompt If branch"
    )
    return branches[0]


def raw_user_message_assignments(branch):
    assignments = []
    for node in ast.walk(branch):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or not isinstance(node.value, ast.List):
            continue
        if not assignment_targets(node):
            continue
        for element in node.value.elts:
            if not isinstance(element, ast.Dict):
                continue
            role = dict_value(element, "role")
            content = dict_value(element, "content")
            if (
                isinstance(role, ast.Constant)
                and role.value == "user"
                and is_sample_subscript(content, "prompt")
            ):
                assignments.append(node)
    return assignments


def is_chat_template_call(node, message_names):
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "apply_chat_template"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "tokenizer"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "self"
    ):
        return False
    uses_messages = any(
        isinstance(argument, ast.Name) and argument.id in message_names
        for argument in [*node.args, *(keyword.value for keyword in node.keywords)]
    )
    has_tokenize_false = any(
        keyword.arg == "tokenize"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is False
        for keyword in node.keywords
    )
    return uses_messages and has_tokenize_false


def chat_template_assignments(branch, message_names):
    return [
        node
        for node in ast.walk(branch)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and assignment_targets(node)
        and is_chat_template_call(node.value, message_names)
    ]


def prompt_dict_returns(branch, rendered_names):
    returns = []
    for node in ast.walk(branch):
        if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Dict):
            continue
        prompt_value = dict_value(node.value, "prompt")
        if isinstance(prompt_value, ast.Name) and prompt_value.id in rendered_names:
            returns.append(node)
    return returns


def conversation_accesses(statements):
    return [
        node
        for statement in statements
        for node in ast.walk(statement)
        if is_sample_subscript(node, "conversations")
        or (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "sample"
            and node.attr == "conversations"
        )
    ]


def raw_prompt_fallbacks(getitem, branch):
    """Find conversations fallback accesses only in the raw branch's else/siblings."""
    fallback_statements = list(branch.orelse)
    for node in ast.walk(getitem):
        for field in ("body", "orelse", "finalbody"):
            statements = getattr(node, field, None)
            if not isinstance(statements, list) or branch not in statements:
                continue
            fallback_statements.extend(statements[statements.index(branch) + 1 :])
    return conversation_accesses(fallback_statements)


def is_torch_cuda_synchronize(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "synchronize"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "cuda"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "torch"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "device"
    )


def is_time_perf_counter(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "perf_counter"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "time"
    )


def is_model_generate_call(node):
    return is_attribute_call(node, "model", "generate")


def timing_region(evaluate_speed, path):
    """Locate the timed loop by its timer → generate → sync → elapsed ordering."""
    for loop in [node for node in ast.walk(evaluate_speed) if isinstance(node, ast.For)]:
        starts = assignments_with_value(loop, is_time_perf_counter)
        generate_calls = [node for node in ast.walk(loop) if is_model_generate_call(node)]
        synchronizes = [node for node in ast.walk(loop) if is_torch_cuda_synchronize(node)]
        start_names = {target for start in starts for target in assignment_targets(start)}
        elapsed = assignments_with_value(
            loop,
            lambda value: any(is_time_perf_counter(node) for node in ast.walk(value))
            and bool(expression_names(value) & start_names),
        )
        for start in starts:
            for generate_call in generate_calls:
                for synchronize in synchronizes:
                    for elapsed_assignment in elapsed:
                        if (
                            start.lineno < generate_call.lineno < synchronize.lineno
                            < elapsed_assignment.lineno
                        ):
                            return loop, start, generate_call, synchronize, elapsed_assignment
    assert False, (
        f"{path}: evaluate_speed needs a loop ordered perf_counter → generate → "
        "torch.cuda.synchronize(device) → elapsed calculation"
    )


def is_batch_list_expression(expression):
    if (
        isinstance(expression, ast.BinOp)
        and isinstance(expression.op, ast.Mult)
        and "batch_size" in expression_names(expression)
    ):
        return isinstance(expression.left, (ast.List, ast.ListComp)) or isinstance(
            expression.right, (ast.List, ast.ListComp)
        )
    if not isinstance(expression, ast.ListComp):
        return False
    return any(
        is_named_call(generator.iter, "range")
        and "batch_size" in expression_names(generator.iter)
        for generator in expression.generators
    )


def assignments_with_value(scope, predicate):
    return [
        node
        for node in ast.walk(scope)
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and predicate(node.value)
    ]


def direct_tokenizer_call(expression, batch_names, assignments):
    calls = [
        node
        for node in ast.walk(expression)
        if is_named_call(node, "tokenizer")
        and node.args
        and depends_on(node.args[0], batch_names, assignments)
    ]
    return calls[0] if len(calls) == 1 else None


def layer_dispatch_loop(model_forward, block_forward, block_position, path):
    loops = [
        node
        for node in ast.walk(model_forward)
        if isinstance(node, ast.For)
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id in loop_target_names(node)
            and bound_argument(call, block_forward, block_position) is not None
            for call in ast.walk(node)
        )
    ]
    assert len(loops) == 1, (
        f"{path}: MiniMindModel.forward must have one layer loop that dispatches MiniMindBlock"
    )
    return loops[0]


def loop_target_names(loop):
    return {node.id for node in ast.walk(loop.target) if isinstance(node, ast.Name)}


def is_existence_call(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"exists", "isfile", "is_file"}
    )


def is_missing_existence_predicate(expression, checkpoint_assignments):
    if isinstance(expression, ast.UnaryOp) and isinstance(expression.op, ast.Not):
        return is_existence_call(expression.operand) and depends_on(
            expression.operand, {"checkpoint_path"}, checkpoint_assignments
        )
    if (
        isinstance(expression, ast.Compare)
        and len(expression.ops) == 1
        and isinstance(expression.ops[0], (ast.Eq, ast.Is))
        and len(expression.comparators) == 1
        and isinstance(expression.comparators[0], ast.Constant)
        and expression.comparators[0].value is False
    ):
        return is_existence_call(expression.left) and depends_on(
            expression.left, {"checkpoint_path"}, checkpoint_assignments
        )
    return False


def resolves_to_true(expression, assignments, seen=None):
    seen = set() if seen is None else seen
    if isinstance(expression, ast.Constant):
        return expression.value is True
    if isinstance(expression, ast.Name) and expression.id not in seen:
        return any(
            resolves_to_true(value, assignments, seen | {expression.id})
            for value in assignments.get(expression.id, [])
        )
    return False


def test_ppo_steps_each_optimizer_and_scheduler_inside_ppo_epoch_loop():
    path = "trainer/train_ppo.py"
    update = lookup_function(parse_module(path), path, "ppo_update")
    loops = [
        node
        for node in ast.walk(update)
        if isinstance(node, ast.For)
        and is_named_call(node.iter, "range")
        and len(node.iter.args) == 1
        and isinstance(node.iter.args[0], ast.Attribute)
        and node.iter.args[0].attr == "ppo_epochs"
        and isinstance(node.iter.args[0].value, ast.Name)
        and node.iter.args[0].value.id == "args"
    ]
    assert len(loops) == 1, f"{path}: ppo_update must contain one range(args.ppo_epochs) loop"

    required_steps = {
        ("actor_optimizer", "step"),
        ("critic_optimizer", "step"),
        ("actor_scheduler", "step"),
        ("critic_scheduler", "step"),
    }
    actual_steps = {
        required
        for required in required_steps
        if any(is_attribute_call(node, *required) for node in ast.walk(loops[0]))
    }
    assert actual_steps == required_steps, (
        f"{path}: all actor/critic optimizer and scheduler steps belong inside PPO epoch loop"
    )
    for optimizer_name, scheduler_name in (
        ("actor_optimizer", "actor_scheduler"),
        ("critic_optimizer", "critic_scheduler"),
    ):
        optimizer_steps = sorted(
            node.lineno
            for node in ast.walk(loops[0])
            if is_attribute_call(node, optimizer_name, "step")
        )
        scheduler_steps = sorted(
            node.lineno
            for node in ast.walk(loops[0])
            if is_attribute_call(node, scheduler_name, "step")
        )
        assert len(optimizer_steps) == len(scheduler_steps), (
            f"{path}: {optimizer_name}.step and {scheduler_name}.step must pair per PPO epoch"
        )
        assert all(
            optimizer_line < scheduler_line
            for optimizer_line, scheduler_line in zip(optimizer_steps, scheduler_steps)
        ), f"{path}: each {optimizer_name}.step must precede {scheduler_name}.step"


def test_rope_indexes_cos_sin_and_threads_positions_through_transformer_layers():
    path = "model/MiniMindModel.py"
    module = parse_module(path)
    rope = lookup_function(module, path, "apply_rotary_pos_emb")
    attention = lookup_class(module, path, "Attention")
    attention_forward = lookup_function(attention, path, "forward")
    block = lookup_class(module, path, "MiniMindBlock")
    block_forward = lookup_function(block, path, "forward")
    model = lookup_class(module, path, "MiniMindModel")
    model_forward = lookup_function(model, path, "forward")

    rope_position = position_parameter(rope, path, "apply_rotary_pos_emb")
    indexed_rope_values = {"cos": set(), "sin": set()}
    for node in ast.walk(rope):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or not isinstance(node.value, ast.Subscript):
            continue
        source = node.value.value
        if not isinstance(source, ast.Name) or source.id not in indexed_rope_values:
            continue
        if not depends_on(node.value.slice, {rope_position}, assignment_map(rope)):
            continue
        indexed_rope_values[source.id].update(assignment_targets(node))

    assert indexed_rope_values["cos"], f"{path}: RoPE must index cos by position ids"
    assert indexed_rope_values["sin"], f"{path}: RoPE must index sin by position ids"

    rope_assignments = assignment_map(rope)
    returns = [node for node in rope.body if isinstance(node, ast.Return)]
    assert returns, f"{path}: apply_rotary_pos_emb must return rotated q and k"
    output_pair = returns[-1].value
    assert isinstance(output_pair, (ast.Tuple, ast.List)) and len(output_pair.elts) == 2, (
        f"{path}: apply_rotary_pos_emb return contract is (q_rotated, k_rotated)"
    )
    for tensor_name, output in zip(("q", "k"), output_pair.elts):
        assert depends_on(output, {tensor_name}, rope_assignments), (
            f"{path}: rotated {tensor_name} output must depend on {tensor_name}"
        )
        for indexed_names in indexed_rope_values.values():
            assert any(
                depends_on(output, {indexed_name}, rope_assignments)
                for indexed_name in indexed_names
            ), f"{path}: rotated {tensor_name} output must depend on indexed RoPE values"

    attention_position = position_parameter(attention_forward, path, "Attention.forward")
    block_position = position_parameter(block_forward, path, "MiniMindBlock.forward")
    model_position = position_parameter(model_forward, path, "MiniMindModel.forward")
    attention_calls = [
        node
        for node in ast.walk(block_forward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "attention"
    ]
    assert attention_calls, f"{path}: MiniMindBlock.forward must call Attention.forward"
    block_assignments = assignment_map(block_forward)
    assert any(
        (argument := bound_argument(call, attention_forward, attention_position)) is not None
        and depends_on(argument, {block_position}, block_assignments)
        for call in attention_calls
    ), f"{path}: MiniMindBlock must bind its position ids into Attention.forward"

    dispatch_loop = layer_dispatch_loop(
        model_forward, block_forward, block_position, path
    )
    layer_names = loop_target_names(dispatch_loop)
    block_calls = [
        node
        for node in ast.walk(dispatch_loop)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in layer_names
        and bound_argument(node, block_forward, block_position) is not None
    ]
    assert block_calls, (
        f"{path}: model layer-loop variable must dispatch MiniMindBlock.forward"
    )
    model_assignments = assignment_map(model_forward)
    assert any(
        (
            isinstance(
                argument := bound_argument(call, block_forward, block_position), ast.Name
            )
            and argument.id == model_position
        )
        or depends_on(
            bound_argument(call, block_forward, block_position),
            {model_position},
            model_assignments,
        )
        for call in block_calls
    ), f"{path}: MiniMindModel must bind its position ids into MiniMindBlock.forward"

    rope_calls = [
        node for node in ast.walk(attention_forward) if is_named_call(node, "apply_rotary_pos_emb")
    ]
    assert rope_calls, f"{path}: Attention.forward must call apply_rotary_pos_emb"
    attention_assignments = assignment_map(attention_forward)
    assert any(
        (argument := bound_argument(call, rope, rope_position)) is not None
        and depends_on(argument, {attention_position}, attention_assignments)
        for call in rope_calls
    ), f"{path}: Attention must bind its position ids into apply_rotary_pos_emb"


def test_grpo_per_token_logps_forwards_attention_mask_to_model():
    path = "trainer/train_grpo.py"
    get_logps = lookup_function(parse_module(path), path, "get_per_token_logps")
    mask_parameters = [name for name in function_parameters(get_logps) if "attention" in name]
    assert mask_parameters, f"{path}: get_per_token_logps must accept attention_mask"
    mask_name = mask_parameters[0]
    model_calls = [node for node in ast.walk(get_logps) if is_named_call(node, "mdl")]
    assert model_calls, f"{path}: get_per_token_logps must call mdl"
    assert any(
        any(
            keyword.arg == "attention_mask"
            and depends_on(keyword.value, {mask_name}, assignment_map(get_logps))
            for keyword in call.keywords
        )
        for call in model_calls
    ), f"{path}: get_per_token_logps must forward attention_mask to mdl"


NON_PPO_TRAINERS = [
    "trainer/train_pretrain.py",
    "trainer/train_full_sft.py",
    "trainer/train_lora.py",
    "trainer/train_dpo.py",
    "trainer/train_reason.py",
    "trainer/train_grpo.py",
]


@pytest.mark.parametrize("path", NON_PPO_TRAINERS)
def test_non_ppo_checkpoint_conditions_require_positive_optimizer_step_guard(path):
    module = parse_module(path)
    parents = parent_map(module)
    all_checkpoint_calls = [node for node in ast.walk(module) if is_lm_checkpoint_call(node)]
    checkpoint_calls = [
        node
        for node in all_checkpoint_calls
        if any(
            isinstance(parent, ast.For) and "step" in loop_target_names(parent)
            for parent in enclosing_nodes(node, parents)
        )
    ]
    assert checkpoint_calls, f"{path}: expected an in-loop lm_checkpoint save call"
    for checkpoint_call in checkpoint_calls:
        optimizer_guards = [
            node
            for node in enclosing_nodes(checkpoint_call, parents)
            if isinstance(node, ast.If)
            and has_positive_and_conjunct(node.test, "did_optimizer_step")
        ]
        assert optimizer_guards, (
            f"{path}: lm_checkpoint must have a nearest enclosing positive "
            "did_optimizer_step And guard (outer is_main_process is allowed)"
        )


def test_eval_loader_rejects_an_explicit_missing_checkpoint():
    path = "evals/core/load_model.py"
    loader = lookup_function(parse_module(path), path, "load_model_and_tokenizer")
    loader_assignments = assignment_map(loader)

    def is_missing_checkpoint_condition(condition):
        terms = flattened_and_conjuncts(condition)
        has_requested_path = any(
            not is_missing_existence_predicate(term, loader_assignments)
            and depends_on(term, {"checkpoint_path"}, loader_assignments)
            for term in terms
        )
        has_missing_file = any(
            is_missing_existence_predicate(term, loader_assignments) for term in terms
        )
        return has_requested_path and has_missing_file

    missing_checkpoint_guards = [
        node
        for node in ast.walk(loader)
        if isinstance(node, ast.If)
        and is_missing_checkpoint_condition(node.test)
        and any(
            isinstance(child, ast.Raise)
            and isinstance(child.exc, ast.Call)
            and is_named_call(child.exc, "FileNotFoundError")
            for child in ast.walk(node)
        )
    ]
    assert missing_checkpoint_guards, (
        f"{path}: explicit missing checkpoint_path must raise FileNotFoundError"
    )


def test_rlaif_raw_prompt_branch_renders_and_returns_before_conversation_fallback():
    path = "dataset/llm_dataset.py"
    module = parse_module(path)
    dataset = lookup_class(module, path, "RLAIFDataset")
    getitem = lookup_function(dataset, path, "__getitem__")
    branch = raw_prompt_branch(getitem, path)

    message_assignments = raw_user_message_assignments(branch)
    assert message_assignments, (
        f"{path}: raw prompt branch must build a user message from sample['prompt']"
    )
    message_names = {
        target for assignment in message_assignments for target in assignment_targets(assignment)
    }
    rendered_assignments = chat_template_assignments(branch, message_names)
    assert rendered_assignments, (
        f"{path}: raw prompt branch must render messages with tokenize=False"
    )
    rendered_names = {
        target for assignment in rendered_assignments for target in assignment_targets(assignment)
    }
    returns = prompt_dict_returns(branch, rendered_names)
    assert returns, f"{path}: raw prompt branch must return its rendered prompt"

    fallbacks = raw_prompt_fallbacks(getitem, branch)
    assert fallbacks, (
        f"{path}: raw-prompt If must have conversations access in its else or sibling fallback"
    )
    first_fallback = min(node.lineno for node in fallbacks)
    assert any(
        assignment.lineno < returned.lineno < first_fallback
        for assignment in rendered_assignments
        for returned in returns
    ), f"{path}: raw prompt rendering and return must precede conversations fallback"


def test_init_model_loads_requested_weights_strictly():
    path = "trainer/trainer_utils.py"
    init_model = lookup_function(parse_module(path), path, "init_model")
    init_assignments = assignment_map(init_model)

    def load_state_argument(call):
        positional = [argument for argument in call.args]
        keyword_values = [
            keyword.value
            for keyword in call.keywords
            if keyword.arg not in {"strict", "assign"}
        ]
        return positional + keyword_values

    strict_loads = [
        node
        for node in ast.walk(init_model)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load_state_dict"
        and depends_on(node.func.value, {"model"}, init_assignments)
        and any(
            depends_on(argument, {"weights"}, init_assignments)
            for argument in load_state_argument(node)
        )
        and any(
            keyword.arg == "strict"
            and resolves_to_true(keyword.value, init_assignments)
            for keyword in node.keywords
        )
    ]
    assert strict_loads, (
        f"{path}: init_model must strictly load the requested weights (aliases/keywords allowed)"
    )


def test_lora_resume_uses_total_epoch_steps_for_accumulation_helpers():
    path = "trainer/train_lora.py"
    module = parse_module(path)
    main_guards = [
        node
        for node in module.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    ]
    assert len(main_guards) == 1, f"{path}: expected one __main__ guard"
    main_guard = main_guards[0]
    resume_calls = [
        node
        for node in ast.walk(main_guard)
        if is_named_call(node, "train_epoch")
        and any(
            keyword.arg == "start_step"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "start_step"
            for keyword in node.keywords
        )
    ]
    assert len(resume_calls) == 1, f"{path}: expected one resumed train_epoch call"
    total_steps = resume_calls[0].args[2]
    assert (
        isinstance(total_steps, ast.BinOp)
        and isinstance(total_steps.op, ast.Add)
        and any(
            is_named_call(node, "len")
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "loader"
            for node in ast.walk(total_steps)
        )
        and any(isinstance(node, ast.Name) and node.id == "start_step" for node in ast.walk(total_steps))
    ), f"{path}: resumed LoRA must pass len(loader) + start_step as total epoch steps"


def test_speed_benchmark_batches_inputs_and_times_generation_with_cuda_sync():
    path = "evals/eval_speed.py"
    evaluate_speed = lookup_function(parse_module(path), path, "evaluate_speed")
    loop, start, generate_call, post_generate_sync, elapsed = timing_region(
        evaluate_speed, path
    )

    batch_assignments = assignments_with_value(loop, is_batch_list_expression)
    batch_names = {
        target for assignment in batch_assignments for target in assignment_targets(assignment)
    }
    assert batch_names, f"{path}: evaluate_speed must build a prompt list from batch_size"

    tokenized_assignments = [
        assignment
        for assignment in ast.walk(loop)
        if isinstance(assignment, (ast.Assign, ast.AnnAssign))
        and assignment_targets(assignment)
            and direct_tokenizer_call(
                assignment.value, batch_names, assignment_map(loop)
            )
    ]
    tokenized_names = {
        target for assignment in tokenized_assignments for target in assignment_targets(assignment)
    }
    assert tokenized_names, f"{path}: tokenizer must receive the batch prompt list"

    assert any(
        keyword.arg is None
        and isinstance(keyword.value, ast.Name)
        and keyword.value.id in tokenized_names
        for keyword in generate_call.keywords
    ), f"{path}: model.generate must receive the same tokenized batch as **inputs"

    generated_outputs = {
        target
        for assignment in ast.walk(loop)
        if isinstance(assignment, (ast.Assign, ast.AnnAssign))
        and any(is_model_generate_call(node) for node in ast.walk(assignment.value))
        for target in assignment_targets(assignment)
    }
    loop_assignments = assignment_map(loop)

    def is_batch_scaled_generated_count(value):
        return depends_on(value, {"batch_size"}, loop_assignments) and any(
            depends_on(value, {output}, loop_assignments)
            for output in generated_outputs
        )

    generated_count_assignments = assignments_with_value(
        loop,
        is_batch_scaled_generated_count,
    )
    generated_count_names = {
        target
        for assignment in generated_count_assignments
        for target in assignment_targets(assignment)
    }
    assert generated_count_names, (
        f"{path}: generated-token count must multiply the generated output by batch_size"
    )
    assert any(
        isinstance(node, ast.AugAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "total_gen_tokens"
        and (
            (isinstance(node.value, ast.Name) and node.value.id in generated_count_names)
            or is_batch_scaled_generated_count(node.value)
        )
        for node in ast.walk(loop)
    ), f"{path}: total_gen_tokens must consume the batch-scaled generated-token count"

    synchronizes = [node for node in ast.walk(loop) if is_torch_cuda_synchronize(node)]
    assert len(synchronizes) >= 2, (
        f"{path}: timing loop must execute torch.cuda.synchronize(device) around generation"
    )
    assert any(sync.lineno < start.lineno for sync in synchronizes), (
        f"{path}: CUDA synchronize must execute before the start perf_counter timestamp"
    )
    assert start.lineno < generate_call.lineno, (
        f"{path}: perf_counter start timestamp must precede model.generate"
    )
    assert generate_call.lineno < post_generate_sync.lineno < elapsed.lineno, (
        f"{path}: CUDA synchronize must execute after model.generate and before elapsed timing"
    )
