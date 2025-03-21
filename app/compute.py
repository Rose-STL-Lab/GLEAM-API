print ("test")

from google.cloud import compute_v1, billing_v1
import json
from google.cloud import storage
import yaml
import time
from google.oauth2 import service_account

print ("compute downloaded")
# credentials = service_account.Credentials.from_service_account_file(
#     "C:/Users/00011/Downloads/epistorm-gleam-api-612347bc95a6.json"
# )
# instances_client = compute_v1.InstancesClient(credentials=credentials)
# billing_client = billing_v1.CloudCatalogClient(credentials=credentials)
# storage_client = storage.Client(credentials=credentials)
# images_client = compute_v1.ImagesClient(credentials=credentials)
# billing_credentials = service_account.Credentials.from_service_account_file(
#     "app/secrets/epistorm-gleam-api-bb556a38b678.json"
# )
instances_client = compute_v1.InstancesClient()
billing_client = billing_v1.CloudCatalogClient()
storage_client = storage.Client()
images_client = compute_v1.ImagesClient()

print ("Instance client found")

def estimate_instance_cost(num_gpu, num_cpu, num_ram, hours):
    SKUS = {'GPU' : '88B8-C3ED-03F0', 'CPU' : 'CF4E-A0C7-E3BF', 'RAM' : 'F449-33EC-A5EF'}
    SKU_PRICE = {'GPU': 0.35, 'CPU': 0.02181159, 'RAM': 0.00292353}
    estimate = 0
    
    # List SKUs for Compute Engine
    skus = billing_client.list_skus(parent="services/6F81-5844-456A")
    # Search for the SKU matching the machine type
    for sku in skus:
        if SKUS['RAM'] == sku.sku_id:
            print(f"Found RAM SKU: {sku.name} - {sku.description}")
            print(sku.pricing_info)
            SKU_PRICE['RAM'] = sku.pricing_info[0].pricing_expression.tiered_rates[0].unit_price.nanos / 1e9
        if SKUS['GPU'] == sku.sku_id:
            print(f"Found GPU SKU: {sku.name} - {sku.description}")
            print(sku.pricing_info)
            SKU_PRICE['GPU'] = sku.pricing_info[0].pricing_expression.tiered_rates[0].unit_price.nanos / 1e9
        if SKUS['CPU'] == sku.sku_id:
            print(f"Found CPU SKU: {sku.name} - {sku.description}")
            print(sku.pricing_info)    
            SKU_PRICE['CPU'] = sku.pricing_info[0].pricing_expression.tiered_rates[0].unit_price.nanos / 1e9
            # for pricing_info in sku.pricing_info:
            #     if pricing_info.pricing_expression.usage_unit == "h":  # Hourly pricing
            #         price_per_hour = pricing_info.pricing_expression.tiered_rates[0].unit_price.units
            #         price_nanos = pricing_info.pricing_expression.tiered_rates[0].unit_price.nanos / 1e9
            #         total_price = hours * price_per_hour + price_nanos
            #         print(f"Price per hour: ${total_price:.4f}")

            #         return total_price
    estimate = (SKU_PRICE['GPU'] * num_gpu + SKU_PRICE['CPU'] * num_cpu + SKU_PRICE['RAM'] * num_ram) * hours
    return estimate

def create_instance_with_docker(
    project_id: str,
    zone: str,
    instance_name: str,
    machine_type: str,
    image_family: str,
    image_project: str,
    docker_image: str,
    beta: float,
    epsilon: float,
    simulations: int,
    days: int,
    bucket: str,
    outfile: str
) -> compute_v1.Instance:
    """
    Creates a Compute Engine VM instance with full API access that pulls and runs a Docker container.

    Args:
        project_id: ID or number of the project you want to use.
        zone: Name of the zone you want to check, for example: us-west3-b.
        instance_name: Name of the new instance.
        machine_type: Machine type for the VM, e.g., "e2-medium".
        image_family: Image family for the VM OS, e.g., "debian-12".
        image_project: Google Cloud project hosting the OS image.
        docker_image: Docker container image to pull and run.
        beta: Value for the beta environment variable.
        epsilon: Value for the epsilon environment variable.

    Returns:
        Instance object.
    """
    startup_script = f"""#!/bin/bash
    IMAGE_NAME=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/image_name -H "Metadata-Flavor: Google")
    CONTAINER_PARAM=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/container_param -H "Metadata-Flavor: Google")
    sudo apt-get update
    sudo apt-get install -y docker.io
    sudo gcloud auth configure-docker
    sudo docker pull {docker_image}
    sudo docker run -e BETA={beta} -e EPSILON={epsilon} -e SIMULATIONS={simulations} -e DAYS={days} -e GCS_BUCKET={bucket} -e OUTFILENAME={outfile} {docker_image}""" + """
    zoneMetadata=$(curl "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor:Google")
    IFS=$'/'
    zoneMetadataSplit=($zoneMetadata)
    ZONE="${zoneMetadataSplit[3]}"
    docker run --entrypoint "gcloud" google/cloud-sdk:alpine compute instances delete ${HOSTNAME}  --delete-disks=all --zone=${ZONE}
    """

    metadata_items = [
        compute_v1.Items(key="startup-script", value=startup_script)
    ]

    # Define the disk for the VM using Debian 12 Bookworm First One I Found that worked, Could be changed for different workloads
    initialize_params = compute_v1.AttachedDiskInitializeParams(
        source_image="projects/debian-cloud/global/images/debian-12-bookworm-v20241112",
        disk_size_gb=10,  # Specify 10GB disk size
        disk_type=f"zones/{zone}/diskTypes/pd-balanced" 
    )
    disk = compute_v1.AttachedDisk(
        boot=True,
        auto_delete=True,
        initialize_params=initialize_params
    )

    # Define the network interface with an external IP Important for Downloading 
    access_config = compute_v1.AccessConfig(name="External NAT", type_="ONE_TO_ONE_NAT")
    network_interface = compute_v1.NetworkInterface(
        name="global/networks/default",
        access_configs=[access_config] 
    )

    # Define the instance
    instance_resource = compute_v1.Instance(
        name=instance_name,
        machine_type=f"zones/{zone}/machineTypes/{machine_type}",
        disks=[disk],
        network_interfaces=[network_interface],
        metadata=compute_v1.Metadata(items=metadata_items),
        service_accounts=[
            compute_v1.ServiceAccount(
                email="default",
                scopes=["https://www.googleapis.com/auth/cloud-platform"] # Important for Pulling Docker Image
            )
        ]
    )

    # Insert the instance
    instance_insert_request = compute_v1.InsertInstanceRequest(
        project=project_id,
        zone=zone,
        instance_resource=instance_resource
    )
    operation = instances_client.insert(request=instance_insert_request)
    operation.result()  # Wait for the operation to complete

    return instances_client.get(project=project_id, zone=zone, instance=instance_name)

def create_dummy_instance(
    project_id: str,
    zone: str,
    instance_name: str,
    machine_type: str,
    cpu: int,
    io: int,
    vm: int,
    vm_bytes: str,
    timeout: str,
    timestamp: str,
    service_account: str = None,
    service_account_key: str = None, 
    bucket_name: str = None,
    folder: str = None
) -> str:

    bucket_name = bucket_name or "seir-output-bucket-2"
    folder = folder or "outputdata"

    startup_script = f"""#!/bin/bash
    IMAGE_NAME=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/image_name -H "Metadata-Flavor: Google")
    sudo apt-get update
    sudo apt-get install -y zip
    sudo apt-get install -y docker.io
    sudo gcloud auth configure-docker
    sudo docker run polinux/stress bash
    sudo apt update && sudo apt install stress -y
    stress --cpu {cpu} --io {io} --vm {vm} --vm-bytes {vm_bytes} --timeout {timeout} --verbose
    mkdir data{timestamp}
    sudo gsutil cp -r gs://seir-output-bucket-2/leam_us_data/data data{timestamp}
    zip -r data{timestamp}.zip data{timestamp}
    """

    if service_account and service_account_key:
        service_account_json = json.dumps(service_account_key)
        startup_script += f"""
        export GOOGLE_APPLICATION_CREDENTIALS_ORIG=$(gcloud auth list --format="value(account)")
        echo '{service_account_json}' > service-account-key.json
        gcloud auth activate-service-account {service_account} --key-file=service-account-key.json
        gsutil -o Credentials:gs_service_key_file=service-account-key.json cp -r data{timestamp}.zip gs://{bucket_name}/{folder}/
        gcloud auth revoke
        """
    else:
        startup_script += f"""
        sudo gsutil cp -r data{timestamp}.zip gs://{bucket_name}/{folder}/
        """

    startup_script += """
    export NAME=$(curl -X GET http://metadata.google.internal/computeMetadata/v1/instance/name -H 'Metadata-Flavor: Google')
    export ZONE=$(curl -X GET http://metadata.google.internal/computeMetadata/v1/instance/zone -H 'Metadata-Flavor: Google')
    gcloud --quiet compute instances delete $NAME --zone=$ZONE
    """


    metadata_items = [
        compute_v1.Items(key="startup-script", value=startup_script)
    ]

    # Define the disk for the VM using Debian 12 Bookworm First One I Found that worked, Could be changed for different workloads
    initialize_params = compute_v1.AttachedDiskInitializeParams(
        source_image="projects/debian-cloud/global/images/debian-12-bookworm-v20241112",
        disk_size_gb=30, 
        disk_type=f"zones/{zone}/diskTypes/pd-balanced" 
    )
    disk = compute_v1.AttachedDisk(
        boot=True,
        auto_delete=True,
        initialize_params=initialize_params
    )

    # Define the network interface with an external IP Important for Downloading 
    access_config = compute_v1.AccessConfig(name="External NAT", type_="ONE_TO_ONE_NAT")
    network_interface = compute_v1.NetworkInterface(
        name="global/networks/default",
        access_configs=[access_config] 
    )

    # Define the instance
    instance_resource = compute_v1.Instance(
        name=instance_name,
        machine_type=f"zones/{zone}/machineTypes/{machine_type}",
        disks=[disk],
        network_interfaces=[network_interface],
        metadata=compute_v1.Metadata(items=metadata_items),
        service_accounts=[
            compute_v1.ServiceAccount(
                email="default",
                scopes=["https://www.googleapis.com/auth/cloud-platform"] # Important for Pulling Docker Image
            )
        ]
    )

    # Insert the instance
    instance_insert_request = compute_v1.InsertInstanceRequest(
        project=project_id,
        zone=zone,
        instance_resource=instance_resource
    )
    operation = instances_client.insert(request=instance_insert_request)
    operation.result()  # Wait for the operation to complete

    

    return instances_client.get(project=project_id, zone=zone, instance=instance_name)


def create_instance_with_image(
    project_id: str,
    zone: str,
    instance_name: str,
    machine_type: str,
    source_image: str,
    beta: float,
    epsilon: float,
    simulations: int,
    days: int,
    bucket: str,
    outfile: str
) -> compute_v1.Instance:
    """
    Creates a Compute Engine VM instance with full API access that pulls and runs a Docker container.

    Args:
        project_id: ID or number of the project you want to use.
        zone: Name of the zone you want to check, for example: us-west3-b.
        instance_name: Name of the new instance.
        machine_type: Machine type for the VM, e.g., "e2-medium".
        source_image: Full path to the source image, e.g., "projects/my-project/global/images/my-custom-image".
        docker_image: Docker image to pull and run.
        beta: Value for the beta environment variable.
        epsilon: Value for the epsilon environment variable.
        simulations: Number of simulations to run.
        days: Number of days to simulate.
        bucket: GCS bucket to use.
        outfile: Output file name.

    Returns:
        Instance object.
    """
    startup_script = f"""#!/bin/bash
    cd ..
    cd ..
    cd opt/myapp
    source venv/bin/activate
    gsutil cp gs://testscriptholder/configs/instance-config.yaml /tmp/config.yaml
    export BETA={beta} EPSILON={epsilon} SIMULATIONS={simulations} DAYS={days} GCS_BUCKET={bucket} OUTFILENAME={outfile}
    python main.py""" + """
    gcloud compute instances delete ${HOSTNAME} --delete-disks=all --zone=$(curl -H "Metadata-Flavor:Google" http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F'/' '{print $4}') --quiet
    """


    metadata_items = [
        compute_v1.Items(key="startup-script", value=startup_script)
    ]

    # Define the disk for the VM using the specified custom image
    initialize_params = compute_v1.AttachedDiskInitializeParams(
        source_image=source_image,
        disk_size_gb=10,  # (we can make this a param in the future)
        disk_type=f"zones/{zone}/diskTypes/pd-balanced"
    )
    disk = compute_v1.AttachedDisk(
        boot=True,
        auto_delete=True,
        initialize_params=initialize_params
    )
    access_config = compute_v1.AccessConfig(name="External NAT", type_="ONE_TO_ONE_NAT")
    network_interface = compute_v1.NetworkInterface(
        name="global/networks/default",
        access_configs=[access_config]
    )

    instance_resource = compute_v1.Instance(
        name=instance_name,
        machine_type=f"zones/{zone}/machineTypes/{machine_type}",
        disks=[disk],
        network_interfaces=[network_interface],
        metadata=compute_v1.Metadata(items=metadata_items),
        service_accounts=[
            compute_v1.ServiceAccount(
                email="default",
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        ]
    )
    instance_insert_request = compute_v1.InsertInstanceRequest(
        project=project_id,
        zone=zone,
        instance_resource=instance_resource
    )
    operation = instances_client.insert(request=instance_insert_request)
    operation.result()

    return instances_client.get(project=project_id, zone=zone, instance=instance_name)




def create_instance_and_save_image(
    project_id: str,
    zone: str,
    instance_name: str,
    machine_type: str,
    image_family: str,
    base_image:str,
    image_project: str,
    bucket_name: str,
    folder_name: str,
    custom_image_name: str
) -> compute_v1.Instance:
    """
    Creates a Compute Engine VM instance, sets up a persistent venv, downloads a script and requirements,
    verifies the setup using logs, and saves the instance as a custom image.

    Args:
        project_id: ID or number of the project you want to use.
        zone: Name of the zone you want to check, for example: us-west3-b.
        instance_name: Name of the new instance.
        machine_type: Machine type for the VM, e.g., "e2-medium".
        image_family: Image family for the VM OS, e.g., "debian-12".
        image_project: Google Cloud project hosting the OS image.
        bucket_name: GCP Storage bucket name containing the script and requirements.
        script_name: Name of the script to download from the bucket.
        requirements_name: Name of the requirements file to download from the bucket.
        custom_image_name: Name of the custom image to create.

    Returns:
        Instance object.
    """
    startup_script = f"""#!/bin/bash
sudo apt-get update
sudo mkdir -p /opt/myapp/
sudo chmod -R 755 /opt/myapp/{folder_name} 

sudo gsutil cp -r gs://{bucket_name}/{folder_name} /opt/myapp
"""

    metadata_items = [
        compute_v1.Items(key="startup-script", value=startup_script)
    ]

    if base_image:
        initialize_params = compute_v1.AttachedDiskInitializeParams(
        source_image=base_image,
        disk_size_gb=50,
        disk_type=f"zones/{zone}/diskTypes/pd-balanced"
    )
    else:
        initialize_params = compute_v1.AttachedDiskInitializeParams(
        source_image=f"projects/{image_project}/global/images/family/{image_family}",
        disk_size_gb=50,
        disk_type=f"zones/{zone}/diskTypes/pd-balanced"
        )
    disk = compute_v1.AttachedDisk(
        boot=True,
        auto_delete=True,
        initialize_params=initialize_params
    )

    access_config = compute_v1.AccessConfig(name="External NAT", type_="ONE_TO_ONE_NAT")
    network_interface = compute_v1.NetworkInterface(
        name="global/networks/default",
        access_configs=[access_config]
    )

    instance_resource = compute_v1.Instance(
        name=instance_name,
        machine_type=f"zones/{zone}/machineTypes/{machine_type}",
        disks=[disk],
        network_interfaces=[network_interface],
        metadata=compute_v1.Metadata(items=metadata_items),
        service_accounts=[
            compute_v1.ServiceAccount(
                email="default",
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        ]
    )

    instance_insert_request = compute_v1.InsertInstanceRequest(
        project=project_id,
        zone=zone,
        instance_resource=instance_resource
    )
    operation = instances_client.insert(request=instance_insert_request)
    operation.result() 
    time.sleep(180)

    stop_request = compute_v1.StopInstanceRequest(
        project=project_id,
        zone=zone,
        instance=instance_name
    )
    stop_operation = instances_client.stop(request=stop_request)
    stop_operation.result()  

    image_request = compute_v1.InsertImageRequest(
        project=project_id,
        image_resource=compute_v1.Image(
            name=custom_image_name,
            source_disk=f"projects/{project_id}/zones/{zone}/disks/{instance_name}"
        )
    )
    image_operation = images_client.insert(request=image_request)
    image_operation.result() 

    delete_request = compute_v1.DeleteInstanceRequest(
        project = project_id,
        instance=instance_name,
        zone=zone
    )

    instances_client.delete(request=delete_request)
    stop_operation.result()

    return custom_image_name


def upload_yaml_to_gcs(yaml_data: dict, bucket_name: str, destination_blob_name: str):
    """
    Uploads a YAML object to a specified GCS bucket.

    Args:
        yaml_data (dict): The parsed YAML data.
        bucket_name (str): The name of the GCS bucket.
        destination_blob_name (str): The destination path in the bucket (e.g., 'folder/file.yaml').

    Returns:
        str: A message indicating success.
    """

    yaml_string = yaml.dump(yaml_data, default_flow_style=False)

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_string(yaml_string, content_type="application/x-yaml")

    return f"File {destination_blob_name} uploaded to {bucket_name}."


def create_instance_with_image_config(
    project_id: str,
    zone: str,
    instance_name: str,
    machine_type: str,
    source_image: str,
    script_location: str,
    bucket: str,
    outfile: str,
    config: str

) -> compute_v1.Instance:
    """
    Creates a Compute Engine VM instance with full API access that pulls and runs a Docker container.

    Args:
        project_id: ID or number of the project you want to use.
        zone: Name of the zone you want to check, for example: us-west3-b.
        instance_name: Name of the new instance.
        machine_type: Machine type for the VM, e.g., "e2-medium".
        source_image: Full path to the source image, e.g., "projects/my-project/global/images/my-custom-image".
        bucket: GCS bucket to use.
        outfile: Output file name.
        config: File name of config

    Returns:
        Instance object.
    """
    startup_script = f"""#!/bin/bash
    cd ..
    cd ..
    cd opt/myapp
    sudo gsutil cp gs://testscriptholder/{config} /opt/myapp/{config}
    source venv/bin/activate
    sudo /opt/myapp/venv/bin/pip install google-cloud-storage
    sudo /opt/myapp/venv/bin/pip install --upgrade pandas
    cat {config}
    cd {script_location}
    export GCS_BUCKET={bucket} OUTFILENAME={outfile}
    python main.py""" + """
    gcloud compute instances delete ${HOSTNAME} --delete-disks=all --zone=$(curl -H "Metadata-Flavor:Google" http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F'/' '{print $4}') --quiet"""


    metadata_items = [
        compute_v1.Items(key="startup-script", value=startup_script)
    ]

    # Define the disk for the VM using the specified custom image
    initialize_params = compute_v1.AttachedDiskInitializeParams(
        source_image=source_image,
        disk_size_gb=50,  # (we can make this a param in the future)
        disk_type=f"zones/{zone}/diskTypes/pd-balanced"
    )
    disk = compute_v1.AttachedDisk(
        boot=True,
        auto_delete=True,
        initialize_params=initialize_params
    )
    access_config = compute_v1.AccessConfig(name="External NAT", type_="ONE_TO_ONE_NAT")
    network_interface = compute_v1.NetworkInterface(
        name="global/networks/default",
        access_configs=[access_config]
    )

    instance_resource = compute_v1.Instance(
        name=instance_name,
        machine_type=f"zones/{zone}/machineTypes/{machine_type}",
        disks=[disk],
        network_interfaces=[network_interface],
        metadata=compute_v1.Metadata(items=metadata_items),
        service_accounts=[
            compute_v1.ServiceAccount(
                email="default",
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        ]
    )
    instance_insert_request = compute_v1.InsertInstanceRequest(
        project=project_id,
        zone=zone,
        instance_resource=instance_resource
    )
    operation = instances_client.insert(request=instance_insert_request)
    operation.result()

    return instances_client.get(project=project_id, zone=zone, instance=instance_name)


